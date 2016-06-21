
#pragma once

#include <boost/serialization/nvp.hpp>
#include <vexcl/vector.hpp>

namespace annlearn {
	using boost::serialization::make_nvp;
}

namespace boost {
namespace serialization {

template <class Archive, typename T>
void save(Archive& ar, const vex::vector<T>& v, const unsigned int version)
{
	std::vector<T> out(v.size());
	vex::copy(v.begin(), v.end(), out.begin());

	ar << boost::serialization::make_nvp("data", out);
}

template <class Archive, typename T>
void load(Archive& ar, vex::vector<T>& v, const unsigned int version)
{
	std::vector<T> in;
	ar >> boost::serialization::make_nvp("data", in);

	v.resize(vex::current_context(), in);
}

template<class Archive, typename T>
inline void serialize(Archive & ar, vex::vector<T>& v, const unsigned int file_version)
{
	split_free(ar, v, file_version);
}

}}

