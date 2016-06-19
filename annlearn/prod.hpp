
#pragma once



template<typename T>
auto vec_mat_prod(const vex::vector<T>& v, const vex::vector<float>& u)
{
	using vex::_;

	vex::slicer<1> vec(vex::extents[v.size()]);
	vex::slicer<2> mat(vex::extents[v.size()][u.size() / v.size()]);

	return vex::tensordot(vec[_](v), mat[_](u), vex::axes_pairs(0, 0));
}

template<typename T>
auto mat_mat_prod(const vex::vector<T>& v, const vex::vector<float>& u, size_t inner_dim)
{
	using vex::_;

	vex::slicer<2> matv(vex::extents[v.size() / inner_dim][inner_dim]);
	vex::slicer<2> matu(vex::extents[inner_dim][u.size() / inner_dim]);

	return vex::tensordot(matv[_](v), matu[_](u), vex::axes_pairs(1, 0));
}
