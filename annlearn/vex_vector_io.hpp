
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


namespace annlearn
{
	
template<typename T>
vex::vector<T> load_vector(std::istream& is)
{
	vex::vector<T> v;
	boost::archive::xml_iarchive ia(is);
	ia >> make_nvp("v", v);

	return v;
}

template<typename T>
void save_vector(std::ostream& os, const vex::vector<T>& v)
{
	boost::archive::xml_oarchive oa(os);
	oa << make_nvp("v", v);
}

}
