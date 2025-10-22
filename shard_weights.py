def shard_linear_column(layer, world_size, rank):
    shard_size = layer.weight.shape[0] // world_size
    return layer.weight.data[rank*shard_size : (rank+1)*shard_size, :].clone()
