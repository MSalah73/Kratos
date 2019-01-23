module Kratos

using Lazy, Random

include("Image.jl")

import .Image

data_dir = "../KratosBackup/deep-fashion"

skip_lines(io::IOStream, n::Integer) = (foreach(_ -> readline(io), 1:n); io)

read_key_values(io::IOStream) = @>> begin
    skip_lines(io, 2)
    eachline()
    map(split)
    Dict{String, String}()
end

eval_partition = open(read_key_values, "$data_dir/eval/list_eval_partition.txt")
category_cloth = open(read_key_values, "$data_dir/anno/list_category_cloth.txt")
category_img = open(read_key_values, "$data_dir/anno/list_category_img.txt")

function partition_dataset(by::AbstractString; shuffle=false)
    images = [f for (f, p) in eval_partition if p == by]
    shuffle && shuffle!(images)
    labels = [category_img[i] for i in images]
    images, labels
end

train = partition_dataset("train", shuffle=true)
val, test = partition_dataset.(["val", "test"])

image = rand(Image.RGB, 350, 250);

transform = Image.Compose(
    Image.Crop(300, 300),
    Image.Pad(300, 300),
    Image.ToChannels(Float32))

end
