module Data

export Dataset, tf, batch, prefetch, repeat

import Base: map, iterate, repeat
import Base.Iterators: zip

using PyCall, Lazy

@pyimport tensorflow as tf

tf.enable_eager_execution()

struct Dataset
    value::PyObject
end

Dataset(data::AbstractArray) =
    Dataset(tf.data["Dataset"]["from_tensor_slices"](data))

map(f, dataset::Dataset; num_parallel_calls=Sys.CPU_THREADS) =
    Dataset(dataset.value["map"](f, num_parallel_calls=num_parallel_calls))

zip(datasets::Vararg{Dataset}) =
    @>> begin
        tuple((dataset.value for dataset in datasets)...)
        tf.data["Dataset"]["zip"]()
        Dataset()
    end

batch(dataset::Dataset, batch_size::Int) =
    Dataset(dataset.value["batch"](batch_size))

prefetch(dataset::Dataset, prefetch_size::Int) =
    Dataset(dataset.value["prefetch"](prefetch_size))

repeat(dataset::Dataset) = Dataset(dataset.value["repeat"]())

unbox(x) = x["numpy"]()
unbox(xs::Tuple) = tuple((x["numpy"]() for x in xs)...)

function iterate(dataset::Dataset)
    x = iterate(dataset.value)
    isnothing(x) && return nothing
    unbox(x[1]), x[2]
end

function iterate(dataset::Dataset, state)
    x = iterate(dataset.value, state)
    isnothing(x) && return nothing
    unbox(x[1]), x[2]
end

end
