
#pragma once

namespace annlearn
{

template<typename Container>
class neuron
{

public:
	template<typename Reductor>
	void update(const Container& inputs, Reductor&& reductor)
	{
		value_ = reductor(inputs * weights_);
	}

private:
	float value_;
	Container weights_;
};

}
