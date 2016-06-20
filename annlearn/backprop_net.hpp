
#pragma once

#include "layer.hpp"


namespace annlearn
{

template<typename T>
class backprop_net
{
public:
	backprop_net()
	{}

	backprop_net(const std::vector<vex::backend::command_queue>& queue, std::vector<size_t> layer_sizes)
	{
		assert(queue.size() == 1 /* multi-device contexts not supported*/ );

		for (size_t i = 1; i < layer_sizes.size(); ++i)
			layers_.emplace_back(queue, layer_sizes[i], layer_sizes[i - 1]);
	}

	void reset(const std::vector<vex::backend::command_queue>& queue, std::vector<size_t> layer_sizes)
	{
		assert(queue.size() == 1 /* multi-device contexts not supported*/ );

		layers_.clear();

		for (size_t i = 1; i < layer_sizes.size(); ++i)
			layers_.emplace_back(queue, layer_sizes[i], layer_sizes[i - 1]);
	}

	void random_initialise()
	{
		for (auto& layer : layers_)
			layer.random_initialise();
	}

	template<typename W>
	void set_weights(size_t layer, W&& weights)
	{
		layers_[layer].set_weights(weights);
	}

	template<typename E>
	const auto& forward_pass(const E& in)
	{
		layers_.front().activate(in);

		for (size_t i = 1; i < layers_.size(); ++i)
			layers_[i].activate(layers_[0].activation);

		return layers_.back().activation;;
	}

	template<typename EI, typename ET>
	void backward_pass(T eta, const EI& in, const ET& target)
	{
		layers_.back().compute_deltas(target);
		
		for (size_t i = layers_.size() - 1; i > 0; --i)
			layers_[i-1].compute_deltas(layers_[i]);

		layers_.front().update_weights(eta, in);

		for (size_t i = 1; i < layers_.size(); ++i)
			layers_[i].update_weights(eta, layers_[i-1].activation);
	}

	size_t num_layers() const { return layers_.size(); }

private:
	std::vector<layer<T>> layers_;
}; 

}
