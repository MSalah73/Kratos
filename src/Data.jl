module Data

using Distributed

import Base: getindex, length

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
    dataset::D
end

function getindex(b::Batched, i::Integer)
    lower = (i - 1) * b.batch_size + 1
    upper = min(lower+b.batch_size-1, length(b.dataset))
    cat(b.dataset[lower:upper]..., dims=4)
end

length(b::Batched) = ceil(Int, length(b.dataset) / b.batch_size)

end
