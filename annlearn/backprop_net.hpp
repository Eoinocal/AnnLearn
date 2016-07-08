
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

	const vex::vector<T>& forward_pass(const vex::vector<T>& in)
	{
		layers_.front().activate(in);

		for (size_t i = 1; i < layers_.size(); ++i)
			layers_[i].activate(layers_[0].activation);

		return layers_.back().activation;
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

	void fit(const matrix<T>& x_train, const matrix<T>& y_train, T eta = 0.05f,  int epochs = 1000)
	{
		std::vector<size_t> indices(x_train.nrow());
		std::iota(indices.begin(), indices.end(), 0);

		boost::progress_display show_progress(epochs);
		
		for (int i = 0; i < epochs; ++i)
		{
			std::shuffle(indices.begin(), indices.end(), std::mt19937{std::random_device{}()});

			for (size_t j : indices)
			{
				forward_pass(x_train.row(j));
				backward_pass(eta, x_train.row(j), y_train.row(j));
			}

			++show_progress;
		}
	}

	template<typename T>
	matrix<T> predict(const matrix<T>& input)
	{
		matrix<T> output{input.data.queue_list(), layers_.back().output_size(), input.nrow()};
		
		std::vector<size_t> indices(input.nrow());
		std::iota(indices.begin(), indices.end(), 0);
		
		boost::progress_display show_progress(static_cast<unsigned long>(indices.size()));
				
		for (size_t j : indices)
		{
			output.row(j) = forward_pass(input.row(j));

			++show_progress;
		}

		return output;
	}

	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive & ar, const unsigned int version)
	{
		ar & make_nvp("layers", layers_);
	}

private:
	std::vector<layer<T>> layers_;
}; 

}
