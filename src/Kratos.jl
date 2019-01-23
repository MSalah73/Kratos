module Kratos

using Lazy, Random, Flux, CuArrays

include("Image.jl")
include("Data.jl")

import .Image, .Data

skip_lines(io::IOStream, n::Integer) = (foreach(_ -> readline(io), 1:n); io)

read_key_values(io::IOStream) = @>> begin
    skip_lines(io, 2)
    eachline()
    map(split)
    Dict{String, String}()
end

data_dir = "deep-fashion"

eval_partition = open(read_key_values, "$data_dir/eval/list_eval_partition.txt")
category_cloth = open(read_key_values, "$data_dir/anno/list_category_cloth.txt")
category_img = open(read_key_values, "$data_dir/anno/list_category_img.txt")

function partition_dataset(by::AbstractString)
    images = [f for (f, p) in eval_partition if p == by]
    shuffle!(images)
    labels = [category_img[i] for i in images]
    images, labels
end

train, val, test = partition_dataset.(["train", "val", "test"])

transform = Image.Compose(
    Image.Crop(300, 300),
    Image.Pad(300, 300),
    Image.ToChannels(Float32))

load_image(filename::AbstractString) =
    transform(Image.load("$data_dir/$filename"))

train_images = @>> begin
    train[1]
    Data.Mapped(load_image)
    Data.Batched(10)
end

image = train_images[1][:, :, :, 1]

Image.ToImage(image)

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
    
    Dense(7200, 4000), BatchNorm(4000, relu), Dropout(0.5),
    Dense(4000, 50),
    softmax) |> gpu

model(gpu(train_images[1]))



end
