
#pragma once

#include "prod.hpp"
#include "fc_layer.hpp"

namespace annlearn
{

template<typename T>
class layer
{
public:
	layer()
	{}

	layer(const std::vector<vex::backend::command_queue> &queue, size_t layer_size, size_t previous_layer) :
		weights{queue, layer_size*previous_layer},
		bias_weights{queue, layer_size},
		activation_(queue, layer_size),
		activation_stale_{true},
		input_size_{previous_layer},
		output_size_{layer_size}
	{
		deltas.resize(activation_.size());
	}

	void random_initialise()
	{
		T scale = 0.01f;

		vex::RandomNormal<T, vex::random::threefry> rnd;
		weights = rnd(vex::element_index(), std::rand()) * scale;
		bias_weights = rnd(vex::element_index(), std::rand()) * scale;
	}
	
	void set_weights(const std::vector<T>& v, const std::vector<T>& b)
	{
		weights.resize(v);
		bias_weights.resize(b);
	}

	void set_weights(const vex::vector<T>& v)
	{
		auto x = activation_.size();
		auto y = weights.size() / x;

		assert(v.size() == (x) * (y + 1));

		vex::slicer<2> slice(vex::extents[y + 1][x]);

		weights.resize(slice[vex::range(0, y)][vex::_](v));
		bias_weights.resize(slice[y][vex::_](v));
	}

	const auto& activate(const vex::vector<T>& input)
	{
		auto net = vec_mat_prod(input, weights) + bias_weights;
		activation_ = ACTIVATION_FN(net);

		activation_stale_ = false;

		return activation_;
	}

	template<typename TT>
	const auto& compute_deltas(const TT& target)
	{
		assert(!activation_stale_);

		auto a = vex::tag<1>(activation_);

		return deltas = (activation_ - target) * ACTIVATION_FN_DX(activation_);
	}

	const auto& compute_deltas(const layer<T>& above)
	{
		assert(!activation_stale_);

		auto a = vex::tag<1>(activation_);

		return deltas = summed_delta(above.deltas.size(), vex::element_index(), vex::raw_pointer(above.deltas), vex::raw_pointer(above.weights))
			* ACTIVATION_FN_DX(activation_);
	}
	
	void update_weights(T eta, const vex::vector<T>& input)
	{
		activation_stale_ = true;

		T lambda = 0.0f;

		weights -= weights*lambda + eta
			* get_delta_x_activation(deltas.size(), vex::element_index(), vex::raw_pointer(deltas), vex::raw_pointer(input));
//			* get_activation(deltas.size(), vex::element_index(), vex::raw_pointer(input)) // NOT a bug that delta's size is used
//			  get_delta(deltas.size(), vex::element_index(), vex::raw_pointer(deltas));

		bias_weights -= bias_weights*lambda + eta * get_delta(deltas.size(), vex::element_index(), vex::raw_pointer(deltas));
	}

	size_t layer_size() const { return activation_.size(); }
	size_t input_size() const { return input_size_; }
	size_t output_size() const { return output_size_; }

	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive & ar, const unsigned int version)
	{
		ar & make_nvp("weights", weights);
		ar & make_nvp("bias_weights", bias_weights);
		ar & make_nvp("activation", activation_);
		ar & make_nvp("deltas", deltas);
		ar & make_nvp("activation_stale", activation_stale_);
		ar & make_nvp("input_size", input_size_);
		ar & make_nvp("output_size", output_size_);
	}

	vex::vector<T> weights;
	vex::vector<T> bias_weights;
	vex::vector<T> deltas;

	const vex::vector<T>& activation() const { return activation_; }

private:
	vex::vector<T> activation_;
	bool activation_stale_;
	size_t input_size_;
	size_t output_size_;
};

}
