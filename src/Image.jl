module Image

using Images

export Pad, Crop

struct Pad
    height::Int
    width::Int
end

function compute_pad(length::Integer, image_length::Integer)
    image_length >= length && return 0, 0
    Δlength = (length - image_length) / 2
    ceil(Int, Δlength), floor(Int, Δlength)
end

function (pad::Pad)(image::AbstractMatrix{<:RGB})
    image_height, image_width = size(image)
    h1, h2 = compute_pad(pad.height, image_height)
    w1, w2 = compute_pad(pad.width, image_width)
    padarray(image, Fill(zero(eltype(image)), (h1, w1), (h2, w2)))
end

struct Crop
    height::Int
    width::Int
end

function compute_crop(length::Integer, image_length::Integer)
    image_length <= length && return 1:image_length
    Δlength = (image_length - length + 1) / 2
    floor(Int, Δlength):(image_length - ceil(Int, Δlength))
end

function (crop::Crop)(image::AbstractMatrix{<:RGB})
    image_height, image_width = size(image)
    h = compute_crop(crop.height, image_height)
    w = compute_crop(crop.width, image_width)
    image[h, w]
end

struct ToChannels{T}
    type::T
end

(c::ToChannels)(image::AbstractMatrix{<:RGB}) =
    convert(Array{c.type, 3}, permuteddimsview(channelview(image), (2, 3, 1)))

Compose(transforms...) =
    x -> reduce((x, transform) -> transform(x), transforms; init=x)

end
