
#pragma once

#include "prod.hpp"

namespace annlearn
{

#define ACTIVATION_FN leaky_relu
#define ACTIVATION_FN_DX leaky_relu_dx

#define NET_FN prod_activation


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
	return (1 - x) * (1 + x);
);

VEX_FUNCTION(float, leaky_relu, (double, x),
	return x > 0.0 ? x : 0.01*x;
);

VEX_FUNCTION(float, leaky_relu_dx, (double, x),
	return x > 0.0 ? 1.0 : 0.01;
);

VEX_FUNCTION(float, leaky_squared, (double, x),
	return x > 0.0 ? x*x : -0.01*x*x;
);

VEX_FUNCTION(float, leaky_squared_dx, (double, x),
	return x > 0.0 ? 2.0*sqrt(x) : 0.01;
);

VEX_FUNCTION(float, leaky_sqrt, (double, x),
	return x > 0.0 ? sqrt(x) : 0.01;
);

VEX_FUNCTION(float, leaky_sqrt_dx, (double, x),
	return x > 0.0 ? 1.0 *x / (2.0) : 0.01*x;
);

VEX_FUNCTION(float, periodic, (double, x),
	return x > sin(x);
);

VEX_FUNCTION(float, periodic_dx, (double, x),
	return x > cos(x);
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

VEX_FUNCTION(float, prod_activation, (size_t, n)(size_t, m)(size_t, j)(double*, v)(double*, w)(double*, b),

	double sum = b[j];

	for (size_t i = 0; i < m; ++i)
	{
		sum += v[i] * w[i*n + j];
	}

	return sum;
);

VEX_FUNCTION(float, power_activation, (size_t, n)(size_t, m)(size_t, j)(double*, v)(double*, w)(double*, b),

	double sum = 1.0; //  sign(v[i]) * pow(fabs(v[i]), b[j]);

	for (size_t i = 0; i < m; ++i)
	{
		sum += sign(v[i]) * pow(fabs(v[i]), w[i*n + j]);
	}

	return sum;
);

VEX_FUNCTION(float, get_delta_x_input_power, (size_t, n)(size_t, j)(double*, d)(double*, w)(double*, a),
	return d[j % n] * pow(a[j / n], w[j]) * log(a[j / n]);
);

VEX_CONSTANT(one, 1.0);

auto sig = [](auto x)
{
	return one() / (one() + exp(-x));
};

auto sig_dx = [](auto x)
{
	return x * (one() - x);
};


template<typename T>
class fc_layer
{
public:
	fc_layer()
	{}

	fc_layer(const std::vector<vex::backend::command_queue> &queue, size_t layer_size, size_t previous_layer) :
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
		T scale = 0.001f;

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
		auto net = NET_FN(bias_weights.size(), input.size(), vex::element_index(), vex::raw_pointer(input), vex::raw_pointer(weights), vex::raw_pointer(bias_weights));

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

	const auto& compute_deltas(const fc_layer<T>& above)
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
			* get_delta_x_activation(deltas.size(), vex::element_index(), vex::raw_pointer(deltas), vex::raw_pointer(input)); //  , vex::raw_pointer(weights));
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
