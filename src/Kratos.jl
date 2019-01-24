module Kratos

using Lazy, Random, Flux, CuArrays

include("Image.jl")
include("Data.jl")

import .Image, .Data

skip_lines(io::IOStream, n::Integer) = (foreach(_ -> readline(io), 1:n); io)

read_key_values(io::IOStream, T=Dict{String, String}) = @>> begin
    skip_lines(io, 2)
    eachline()
    map(split)
    T()
end

data_dir = "deep-fashion"

eval_partition = open(read_key_values, "$data_dir/eval/list_eval_partition.txt")
category_img = open(read_key_values, "$data_dir/anno/list_category_img.txt")
category_cloth = open("$data_dir/anno/list_category_cloth.txt") do file
    read_key_values(file, Vector)
end

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

encode_label(label::AbstractString) = Flux.onehot(parse(Int, label), 1:50)

decode_label(encoded::AbstractVector) =
    category_cloth[Flux.onecold(encoded, 1:50)][1]


crossentropy(ŷ::AbstractArray, y::AbstractArray; weight = 1) =
  -sum(y .* log.(ŷ) .* weight) * 1 // size(y, 2)

train_images = @>> begin
    train[1]
    Data.Mapped(load_image)
    Data.Batched(10, 4)
end

train_labels = @>> begin
    train[2]
    Data.Mapped(encode_label)
    Data.Batched(10, 2)
end

train_dataset = Data.Zipped(train_images, train_labels)

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

(images, labels), state = iterate(train_dataset, state)

predictions = model(gpu(images))

image = images[:, :, :, 4]

label = labels[:, 4]

prediction = predictions[:, 4]

Image.ToImage(image)

decode_label(label)

decode_label(prediction)

train[2][20]


end
