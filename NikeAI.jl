using Lazy, Random, Flux, CuArrays
using Flux: Params, params
using BSON: @save

cd(@__DIR__)

includet("Data.jl")

import .Data: tf, batch, prefetch

data_folder = "deep-fashion"

skiplines(io::IOStream, n::Integer) = foreach(_ -> readline(io), 1:n)

key_values_from_file(filename::AbstractString;
                     key_fn=String, value_fn=String) =
    open("$data_folder/$filename") do file
        skiplines(file, 2)
        Dict(key_fn(key) => value_fn(value)
             for (key, value) in split.(readlines(file)))
    end

image_to_partition = key_values_from_file("eval/list_eval_partition.txt")

image_to_label = key_values_from_file(
    "anno/list_category_img.txt", value_fn=x -> parse(Int32, x))

category_name_to_type = key_values_from_file("anno/list_category_cloth.txt")

function partition_dataset(by::AbstractString)
    images = [image for (image, partition) in image_to_partition
              if partition == by]
    shuffle!(images)
    labels = [image_to_label[image] for image in images]
    images, labels
end

train, val, test = partition_dataset.(("train", "val", "test"))

load_image(filename) = @> begin
    tf.read_file(data_folder * "/" + filename)
    tf.image["decode_jpeg"]()
    tf.image["resize_image_with_crop_or_pad"](300, 300)
    tf.image["per_image_standardization"]()
end

function dataset_iterator(dataset)
    images = map(load_image, Data.Dataset(dataset[1]))
    labels = map(l -> tf.one_hot(l, 50), Data.Dataset(dataset[2]))

    @as x begin
        zip(images, labels)
        batch(x, 10)
        map((i, l) -> (tf.transpose(i, (1, 2, 3, 0)), tf.transpose(l)), x)
        prefetch(x, 1)
    end
end

train_dataset = dataset_iterator(train)
val_dataset = dataset_iterator(val)
test_dataset = dataset_iterator(val)


model = Chain(
    Conv((3, 3), 3=>8, stride=(2, 2)), BatchNorm(8, relu),
    Conv((3, 3), 8=>16), BatchNorm(16, relu),
    MaxPool((2, 2)),

    Conv((3, 3), 16=>32), BatchNorm(32, relu),
    Conv((3, 3), 32=>32), BatchNorm(32, relu),
    MaxPool((2, 2)),

    Conv((3, 3), 32=>32), BatchNorm(32, relu),
    Conv((3, 3), 32=>32), BatchNorm(32, relu),
    MaxPool((2, 2)),

    x -> reshape(x, :, size(x, 4)),

    Dense(14400, 5000, relu),
    Dropout(0.5),

    Dense(5000, 50),
    softmax) |> gpu


crossentropy(ŷ::AbstractArray, y::AbstractArray; weight=1) =
    -sum(y .* log.(ŷ) .* weight) * 1 // size(y, 2)

loss(x, y) = crossentropy(model(gpu(x)), gpu(y))

optimizer = ADAM(0.0001)

function train!(loss, parameters, dataset, optimizer)
    for (index, data) in enumerate(dataset)
        l = loss(gpu(data)...)
        Flux.back!(l)
        Flux.Optimise.update!(optimizer, parameters)
        index % 10 == 0 && @show l
        index % 100 == 0 && @save "model/checkpoint$index.bson" model
    end
end


train!(loss, params(model), train_dataset, optimizer)
