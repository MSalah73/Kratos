using Lazy, Random, Flux
using Flux: Params, params

includet("Data.jl")

import .Data: tf, batch, prefetch

data_folder = "../nike-ai/deep-fashion"

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
    Conv((5, 5), 3=>4, stride=(2, 2)), BatchNorm(4, relu),
    Conv((5, 5), 4=>4), BatchNorm(4, relu),
    MaxPool((2, 2)),

    x -> reshape(x, :, size(x, 4)),
    Dense(20736, 50),
    softmax
    )

loss(x, y) = Flux.crossentropy(model(Float64.(x)), y)

images, labels = first(train_dataset)

optimizer = ADAM(params(model), 0.1)

# Flux.train!(loss, train_dataset, optimizer)

Flux.Tracker.gradient(() -> loss(images, labels), Params(params(model)))
