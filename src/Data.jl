module Data

using Distributed

import Base: getindex, length, iterate

export Mapped, Batched

struct Mapped{T, D}
    transform::T
    dataset::D
end

getindex(m::Mapped, i::Integer) = m.transform(m.dataset[i])
getindex(m::Mapped, i::UnitRange) = pmap(m.transform, m.dataset[i])

length(m::Mapped) = length(m.dataset)

struct Batched{D}
    batch_size::Int
    dim::Int
    dataset::D
end

function getindex(b::Batched, i::Integer)
    lower = (i - 1) * b.batch_size + 1
    upper = min(lower+b.batch_size-1, length(b.dataset))
    cat(b.dataset[lower:upper]..., dims=b.dim)
end

length(b::Batched) = ceil(Int, length(b.dataset) / b.batch_size)

struct Zipped{A, B}
    a::A
    b::B
end

getindex(z::Zipped, i::Integer) = (z.a[i], z.b[i])

iterate(z::Zipped, state=1) = i > length(z) ? nothing : z[state], state + 1

end
