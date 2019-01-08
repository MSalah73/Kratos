using Lazy, Random

includet("Data.jl")

import .Data: tf

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
end

train_labels = Data.Dataset(train[2])

train_dataset = zip(train_images, train_labels)

first(train_dataset)

first(train_images)

first(train_labels)
