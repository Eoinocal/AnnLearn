
#pragma once

#include "prod.hpp"

namespace annlearn
{

VEX_FUNCTION(float, sigmoid, (double, x),
	return 1 / (1 + exp(-x));
);

VEX_FUNCTION(float, sigmoid_dx, (double, x),
	return x * (1 - x);
);

VEX_FUNCTION(float, hypertan, (double, x),
	return tanh(x);
);

VEX_FUNCTION(float, hypertan_dx, (double, x),
	return max(-0.5, min(0.5, (1.0 - x) * (1.0 + x)));
);

VEX_FUNCTION(float, max_fn, (double, x),
	return max(0.0, x);
);

VEX_FUNCTION(float, max_dx, (double, x),
	return x < 0.0 ? 0.1 : 1.0;
);


VEX_FUNCTION(float, get_delta_x_activation, (size_t, n)(size_t, j)(double*, d)(double*, a),
	return d[j % n] * a[j / n];
);

VEX_FUNCTION(float, get_delta, (size_t, n)(size_t, j)(double*, d),
	return d[j % n];
);

VEX_FUNCTION(float, get_activation, (size_t, n)(size_t, j)(double*, a),
	return a[j / n];
);

VEX_FUNCTION(float, summed_delta, (size_t, n)(size_t, j)(double*, d)(double*, w),

	float sum = 0.0;

	for (size_t i = 0; i < n; ++i)
		sum += d[i] * w[j*n + i];

	return sum;
);

template<typename T>
class layer
{
public:
	layer()
	{}

	layer(const std::vector<vex::backend::command_queue> &queue, size_t layer_size, size_t previous_layer) :
		weights{queue, layer_size*previous_layer},
		bias_weights{queue, layer_size},
		activation(queue, layer_size),
		activation_stale_{true},
		input_size_{previous_layer},
		output_size_{layer_size}
	{
		deltas.resize(activation.size());
	}

	void random_initialise()
	{
		T scale = 0.1f;

		vex::RandomNormal<T, vex::random::threefry> rnd;
		weights = rnd(vex::element_index(), std::rand()) * scale;
		bias_weights = (2 * rnd(vex::element_index(), std::rand()) - 1) * scale;
	}
	
	void set_weights(const std::vector<T>& v, const std::vector<T>& b)
	{
		weights.resize(v);
		bias_weights.resize(b);
	}

	void set_weights(const vex::vector<T>& v)
	{
		auto x = activation.size();
		auto y = weights.size() / x;

		assert(v.size() == (x) * (y + 1));

		vex::slicer<2> slice(vex::extents[y + 1][x]);

		weights.resize(slice[vex::range(0, y)][vex::_](v));
		bias_weights.resize(slice[y][vex::_](v));
	}

	void activate(const vex::vector<T>& input)
	{
		activation_stale_ = false;

		auto net = vec_mat_prod(input, weights) + bias_weights;
		activation = max_fn(net);
	}

	template<typename TT>
	const auto& compute_deltas(const TT& target)
	{
		assert(!activation_stale_);

		return deltas = (activation - target) * max_dx(activation);
	}

	const auto& compute_deltas(layer<T>& above)
	{
		assert(!activation_stale_);

		return deltas = summed_delta(above.deltas.size(), vex::element_index(), vex::raw_pointer(above.deltas), vex::raw_pointer(above.weights))
			* max_dx(activation);
	}
	
	void update_weights(T eta, const vex::vector<T>& input)
	{
		T lambda = 0.0f;

		weights -= weights*lambda + eta
			* get_delta_x_activation(deltas.size(), vex::element_index(), vex::raw_pointer(deltas), vex::raw_pointer(input));
//			* get_activation(deltas.size(), vex::element_index(), vex::raw_pointer(input)) // NOT a bug that delta's size is used
//			  get_delta(deltas.size(), vex::element_index(), vex::raw_pointer(deltas));

		bias_weights -= bias_weights*lambda + eta * get_delta(deltas.size(), vex::element_index(), vex::raw_pointer(deltas));

		activation_stale_ = true;
	}

	size_t layer_size() const { return activation.size(); }
	size_t input_size() const { return input_size_; }
	size_t output_size() const { return output_size_; }

	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive & ar, const unsigned int version)
	{
		ar & make_nvp("weights", weights);
		ar & make_nvp("bias_weights", bias_weights);
		ar & make_nvp("activation", activation);
		ar & make_nvp("deltas", deltas);
		ar & make_nvp("activation_stale", activation_stale_);
		ar & make_nvp("input_size", input_size_);
		ar & make_nvp("output_size", output_size_);
	}

	vex::vector<T> weights;
	vex::vector<T> bias_weights;
	vex::vector<T> activation; 
	vex::vector<T> deltas;

private:
	bool activation_stale_;
	size_t input_size_;
	size_t output_size_;
};

}
