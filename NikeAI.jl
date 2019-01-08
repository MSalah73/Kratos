using Lazy, Random, Flux

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


train_images = @>> begin
    Data.Dataset(train[1])
    map(f -> tf.read_file(data_folder * "/" + f))
    map(tf.image["decode_jpeg"])
    map(i -> tf.image["resize_image_with_crop_or_pad"](i, 300, 300))
    map(tf.image["per_image_standardization"])
end

train_labels = @>> begin
    Data.Dataset(train[2])
    map(l -> tf.one_hot(l, 50))
end

train_dataset = @as x begin
    zip(train_images, train_labels)
    batch(x, 10)
    map((i, l) -> (tf.transpose(i, (1, 2, 3, 0)), tf.transpose(l)), x)
    repeat(x)
    prefetch(x, 1)
end

model = Chain(
    Conv((3, 3), 3=>4, relu, stride=(2, 2)),
    Conv((3, 3), 4=>6, relu),
    MaxPool((2, 2)),

    Conv((3, 3), 6=>8, relu),
    Conv((3, 3), 8=>10, relu),
    MaxPool((2, 2)),

    Conv((3, 3), 10=>10, relu),
    Conv((3, 3), 10=>10, relu),
    MaxPool((2, 2)),

    x -> reshape(x, :, size(x, 4)),

    Dense(2250, 2000, relu),
    Dropout(0.5),

    Dense(2000, 50),
    softmax)

image, label = first(train_dataset)

model(Float64.(image))
