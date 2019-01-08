module Data

export Dataset, tf

import Base: map, iterate
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
