
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

	backprop_net(const std::vector<vex::backend::command_queue>& queue, std::vector<size_t> layer_sizes) :
		input_(queue, layer_sizes.front()),
		output_(queue, layer_sizes.back())
	{
		for (size_t i = 1; i < layer_sizes.size(); ++i)
			layers_.emplace_back(queue, layer_sizes[i], layer_sizes[i - 1], i != layer_sizes.size() - 1);
	}

	void reset(const std::vector<vex::backend::command_queue>& queue, std::vector<size_t> layer_sizes)
	{
		input_.resize(queue, layer_sizes.front());
		output_.resize(queue, layer_sizes.back());

		layers_.clear();

		for (size_t i = 1; i < layer_sizes.size(); ++i)
			layers_.emplace_back(queue, layer_sizes[i], layer_sizes[i - 1], i != layer_sizes.size() - 1);
	}

	void random_initialise()
	{
		for (auto& layer : layers_)
			layer.random_initialise();
	}

	template<typename W>
	void set_weights(size_t layer, W&& weights)
	{
		layers_[layer].weights.resize(weights);
	}

	std::vector<T> forward_pass(const std::vector<float>& in)
	{
		input_.resize(in);
		vex::vector<T> expr = layers_.front().activate(input_);

		for (size_t i = 1; i < layers_.size(); ++i)
			expr.resize(layers_[i].activate(expr));

		vex::vector<T> out = expr;
		
		std::vector<T> us(out.size());
		vex::copy(out.begin(), out.end(), us.begin());
		return us;
	}

	void backward_pass(T eta, const std::vector<float>& in, const std::vector<float>& target)
	{
		output_.resize(target);
		input_.resize(in);

		layers_.back().compute_deltas(vex::vector<T>(output_));

		//std::cout << (layers_.size() - 2) << std::endl;

		for (size_t i = layers_.size() - 1; i > 0; --i)
		{
		//	std::cout << i << std::endl;
			layers_[i-1].compute_deltas(layers_[i]);
		}

		layers_.front().update_weights(eta, input_);

		for (size_t i = 1; i < layers_.size(); ++i)
			layers_[i].update_weights(eta, layers_[i-1].activation);
	}

private:
	vex::vector<T> input_;
	vex::vector<T> output_;
	std::vector<layer<T>> layers_;
};

}
