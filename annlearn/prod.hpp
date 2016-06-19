
#pragma once


template<typename ExprLeft, typename ExprRight>
auto vec_mat_prod(const ExprLeft& v, const ExprRight& u)
{
	using vex::_;

	vex::detail::get_expression_properties v_prop;
	vex::detail::extract_terminals()(v, v_prop);

	vex::detail::get_expression_properties u_prop;
	vex::detail::extract_terminals()(u, u_prop);

	vex::slicer<1> vec(vex::extents[v_prop.size]);
	vex::slicer<2> mat(vex::extents[v_prop.size][u_prop.size / v_prop.size]);

	return vex::tensordot(vec[_](v), mat[_](u), vex::axes_pairs(0, 0));
}

template<typename ExprLeft, typename T>
auto vec_mat_prod(const ExprLeft& v, const vex::vector<T>& u)
{
	using vex::_;

	vex::detail::get_expression_properties v_prop;
	vex::detail::extract_terminals()(v, v_prop);

	vex::slicer<1> vec(vex::extents[v_prop.size]);
	vex::slicer<2> mat(vex::extents[v_prop.size][u.size() / v_prop.size]);

	return vex::tensordot(vec[_](v), mat[_](u), vex::axes_pairs(0, 0));
}

template<typename T>
auto vec_mat_prod(const vex::vector<T>& v, const vex::vector<T>& u)
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

template<typename ExprLeft, typename ExprRight>
auto inner_product(const ExprLeft& l, const ExprRight& r)
{
	vex::detail::get_expression_properties prop;
	vex::detail::extract_terminals()(l, prop);

	vex::Reductor<double, vex::SUM> sum(prop.queue);
	return sum(l * r);
}
