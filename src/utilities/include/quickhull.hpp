/* Quickhull algorithm implementation
 *
 * Copyright (c) 2014-2015, Anatoliy V. Tomilov
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following condition is met:
 * Redistributions of source code must retain the above copyright notice, this condition and the following disclaimer.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */
#pragma once

#include <type_traits>
#include <vector>
#include <deque>
#include <list>
#include <map>
#include <unordered_set>
#include <unordered_map>
#include <iterator>
#include <memory>
#include <algorithm>
#include <numeric>
#include <utility>
#include <functional>

#include <cstdint>
#include <cmath>
#include <cassert>

template< typename point_iterator,
          typename value_type = std::decay_t< decltype(*std::cbegin(std::declval< typename std::iterator_traits< point_iterator >::value_type >())) > >
struct quick_hull
{

    static_assert(std::is_base_of< std::forward_iterator_tag, typename std::iterator_traits< point_iterator >::iterator_category >::value,
                  "multipass guarantee required");

    using size_type = std::size_t;

    size_type const dimension_;
    value_type const & eps;

    value_type const zero = value_type(0);
    value_type const one = value_type(1);

    using vector = std::vector< value_type >;

private :

    using vrow = value_type *;
    using crow = value_type const *;
    using matrix = std::vector< vrow >;

    vector storage_;
    vrow inner_point_;
    matrix matrix_;
    matrix det_matrix_;
    matrix shadow_matrix_;

public :

    quick_hull(size_type, value_type const &&) = delete; // bind eps to lvalue only

    quick_hull(size_type const _dimension,
               value_type const & _eps)
        : dimension_(_dimension)
        , eps(_eps)
        , storage_(dimension_ * dimension_ * 2 + dimension_)
        , inner_point_(storage_.data())
        , matrix_(dimension_)
        , det_matrix_(dimension_)
        , shadow_matrix_(dimension_)
        , vertices_hashes_(dimension_)
    {
        assert(1 < dimension_);
        assert(!(eps < zero));
        for (vrow & row_ : matrix_) {
            row_ = inner_point_;
            inner_point_ += dimension_;
        }
        for (vrow & row_ : shadow_matrix_) {
            row_ = inner_point_;
            inner_point_ += dimension_;
        }
        assert(inner_point_ + dimension_ == &storage_.back() + 1);
    }

    using point_array = std::vector< point_iterator >;
    using point_list  = std::list< point_iterator >;
    using point_deque = std::deque< point_iterator >;
    using facet_array = std::vector< size_type >;

    struct facet // (d - 1)-dimensional face
    {

        // each neighbouring facet lies against corresponding vertex and vice versa
        point_array vertices_; // dimension_ points (oriented)
        facet_array neighbours_; // dimension_ neighbouring facets

        point_list outside_; // if empty, then is convex hull's facet, else the first point (i.e. outside_.front()) is the furthest point from this facet
        point_deque coplanar_; // containing coplanar points and vertices of coplanar facets as well

        // equation of supporting hyperplane
        vector normal_; // components of normalized normal vector
        value_type D; // distance from the origin to the hyperplane

        template< typename iterator >
        value_type
        distance(iterator const _point) const
        {
            using iterator_traits = std::iterator_traits< iterator >;
            static_assert(std::is_base_of< std::input_iterator_tag, typename iterator_traits::iterator_category >::value);
            return std::inner_product(std::cbegin(normal_), std::cend(normal_), _point, D);
        }

    };

    using facets = std::deque< facet >;

    facets facets_;

    value_type
    cos_of_dihedral_angle(facet const & _first, facet const & _second) const
    {
        return std::inner_product(std::cbegin(_first.normal_), std::cend(_first.normal_), std::cbegin(_second.normal_), zero);
    }

private :

    void
    make_facet(facet & _facet,
               point_array const & _vertices,
               size_type const _against,
               point_iterator const _apex,
               size_type const _neighbour)
    {
        assert(_vertices.size() == dimension_);
        _facet.vertices_ = _vertices;
        _facet.vertices_[_against] = _apex;
        _facet.neighbours_.resize(dimension_);
        _facet.neighbours_[_against] = _neighbour;
        _facet.normal_.resize(dimension_);
    }

    template< typename iterator >
    void
    make_facet(facet & _facet,
               iterator sbeg, // simplex
               size_type const _vertex,
               bool const _swap)
    {
        using iterator_traits = std::iterator_traits< iterator >;
        static_assert(std::is_base_of< std::input_iterator_tag, typename iterator_traits::iterator_category >::value);
        static_assert(std::is_constructible< point_iterator, typename iterator_traits::value_type >::value);
        _facet.vertices_.reserve(dimension_);
        _facet.neighbours_.reserve(dimension_);
        for (size_type v = 0; v <= dimension_; ++v) {
            if (v != _vertex) {
                _facet.vertices_.push_back(*sbeg);
                _facet.neighbours_.push_back(v);
            }
            ++sbeg;
        }
        if (_swap == (((dimension_ - _vertex) % 2) == 0)) {
            using std::swap;
            swap(_facet.vertices_.front(), _facet.vertices_.back());
            swap(_facet.neighbours_.front(), _facet.neighbours_.back());
        }
        _facet.normal_.resize(dimension_);
    }

    void
    reuse_facet(facet & _facet,
                point_array const & _vertices,
                size_type const _against,
                point_iterator const _apex,
                size_type const _neighbour)
    {
        assert(_vertices.size() == dimension_);
        _facet.vertices_ = _vertices;
        _facet.vertices_[_against] = _apex;
        assert(_facet.neighbours_.size() == dimension_);
        _facet.neighbours_[_against] = _neighbour;
        assert(_facet.normal_.size() == dimension_);
    }

    void
    copy_point(point_iterator const _from, vrow _to) const
    {
        std::copy_n(std::cbegin(*_from), dimension_, _to);
    }

    void
    subtract(vrow _minuend, crow _subtrahend) const
    {
        for (size_type i = 0; i < dimension_; ++i) {
            *_minuend++ -= *_subtrahend++;
        }
    }

    void
    gshift(vrow _augend, value_type const & _addend) const // shift Gaussian row
    {
        for (size_type i = 0; i < dimension_; ++i) {
            *_augend++ += _addend;
        }
        *_augend += _addend;
    }

    void
    divide(vrow _dividend, value_type const & _divisor) const
    {
        for (size_type i = 0; i < dimension_; ++i) {
            *_dividend++ /= _divisor;
        }
    }

    void
    subtract_and_assign(vrow _assignee, vrow _minuend, crow _subtrahend) const
    {
        for (size_type i = 0; i < dimension_; ++i) {
            *_assignee++ = (*_minuend++ -= *_subtrahend++);
        }
    }

    void
    multiply_and_add(vrow _assignee, crow _multiplicand, value_type const & _factor) const
    {
        for (size_type i = 0; i < dimension_; ++i) {
            *_assignee++ += (*_multiplicand++ * _factor);
        }
    }

    void
    scale_and_shift(vrow _multiplicand, crow _direction, value_type const & _factor) const
    {
        for (size_type i = 0; i < dimension_; ++i) {
            (*_multiplicand++ *= _factor) += *_direction++;
        }
    }

    void
    matrix_transpose_copy(point_array const & _vertices)
    {
        for (size_type r = 0; r < dimension_; ++r) {
            auto v = std::cbegin(*_vertices[r]);
            for (size_type c = 0; c < dimension_; ++c) {
                shadow_matrix_[c][r] = *v;
                ++v;
            }
        }
    }

    void
    matrix_restore(size_type const _identity)
    {
        for (size_type c = 0; c < _identity; ++c) {
            std::copy_n(shadow_matrix_[c], dimension_, matrix_[c]);
        }
        std::fill_n(matrix_[_identity], dimension_, one);
        for (size_type c = _identity + 1; c < dimension_; ++c) {
            std::copy_n(shadow_matrix_[c], dimension_, matrix_[c]);
        }
    }

    void
    matrix_restore()
    {
        for (size_type c = 0; c < dimension_; ++c) {
            std::copy_n(shadow_matrix_[c], dimension_, matrix_[c]);
        }
    }

    void
    matrix_sqr(size_type const _size)
    { // shadow_matrix_ = matrix_ * transposed matrix_
        assert(_size < dimension_);
        for (size_type r = 0; r < _size; ++r) {
            vrow const lhs_ = shadow_matrix_[r];
            crow const row_ = matrix_[r];
            for (size_type c = 0; c < _size; ++c) {
                lhs_[c] = std::inner_product(row_, row_ + _size, matrix_[c], zero);
            }
        }
    }

    // based on LUP decomposition (complexity is (d^3 / 3 + d^2 / 2 - 5 * d / 6) vs (2 * d^3 / 3 + d^2 + d / 3 - 2) for QR decomposition via Householder reflections)
    value_type
    det(matrix const & _matrix, size_type const _dimension) // hottest function (52% of runtime for D10)
    { // det_matrix_ contains lower unit triangular matrix and upper triangular at return
        assert(0 < _dimension);
        value_type det_ = one;
        std::copy_n(std::cbegin(_matrix), _dimension, std::begin(det_matrix_));
        for (size_type i = 0; i < _dimension; ++i) {
            vrow & mi_ = det_matrix_[i];
            size_type pivot = i;
            {
                using std::abs;
                value_type max_ = abs(mi_[i]);
                size_type j = i;
                while (++j < _dimension) {
                    value_type y_ = abs(det_matrix_[j][i]);
                    if (max_ < y_) {
                        max_ = std::move(y_);
                        pivot = j;
                    }
                }
                if (!(eps < max_)) { // regular?
                    det_ = zero; // singular
                    break;
                }
            }
            if (pivot != i) {
                det_ = -det_; // each permutation flips sign of det
                std::swap(mi_, det_matrix_[pivot]);
            }
            value_type const & dia_ = mi_[i];
            det_ *= dia_; // det is multiple of diagonal elements
            size_type j = i;
            while (++j < _dimension) {
                vrow const mj_ = det_matrix_[j];
                value_type & mji_ = mj_[i];
                mji_ /= dia_;
                size_type k = i;
                while (++k < _dimension) {
                    mj_[k] -= mji_ * mi_[k];
                }
            }
        }
        return det_;
    }

    value_type
    det()
    {
        return det(matrix_, dimension_);
    }

    void
    set_hyperplane_equation(facet & _facet)
    {
        matrix_transpose_copy(_facet.vertices_);
        matrix_restore();
        _facet.D = -det();
        value_type N = zero;
        for (size_type i = 0; i < dimension_; ++i) {
            matrix_restore(i);
            value_type & n = _facet.normal_[i];
            n = det();
            N += n * n;
        }
        using std::sqrt;
        N = sqrt(std::move(N));
        divide(_facet.normal_.data(), N);
        _facet.D /= std::move(N);
        assert(_facet.distance(inner_point_) < zero);
    }

    bool
    orthonormalize(point_list const & _affine_space,
                   size_type const _rank,
                   crow const _origin)
    {
        assert(!(dimension_ < _rank));
        assert(!(_affine_space.size() < _rank));
        auto vertex = std::begin(_affine_space);
        for (size_type r = 0; r < _rank; ++r) { // affine space -> vector space
            vrow const row_ = shadow_matrix_[r];
            copy_point(*vertex, row_);
            subtract(row_, _origin);
            ++vertex;
        }
        for (size_type i = 0; i < _rank; ++i) { // Householder transformation
            value_type sum_ = zero;
            vrow const qri_ = shadow_matrix_[i];
            for (size_type k = i; k < dimension_; ++k) {
                value_type const & qrik_ = qri_[k];
                sum_ += qrik_ * qrik_;
            }
            using std::sqrt;
            value_type norm_ = sqrt(sum_);
            if (!(eps < norm_)) {
                return false;
            }
            value_type & qrii_ = qri_[i];
            if (qrii_ < zero) {
                norm_ = -norm_;
            }
            value_type factor_ = sqrt(std::move(sum_) + qrii_ * norm_);
            if (!(eps < factor_)) {
                return false;
            }
            qrii_ += std::move(norm_);
            for (size_type k = i; k < dimension_; ++k) {
                qri_[k] /= factor_;
            }
            size_type j = i;
            while (++j < _rank) {
                vrow const qrj_ = shadow_matrix_[j];
                value_type s_ = zero;
                for (size_type k = i; k < dimension_; ++k) {
                    s_ += qri_[k] * qrj_[k];
                }
                for (size_type k = i; k < dimension_; ++k) {
                    qrj_[k] -= qri_[k] * s_;
                }
            }
        } // shadow_matrix_ is packed QR
        return true;
    }

    void
    forward_transformation(size_type const _rank) // calculation of Q
    {
        assert(!(dimension_ < _rank));
        for (size_type i = 0; i < _rank; ++i) {
            vrow const qi_ = matrix_[i];
            std::fill_n(qi_, dimension_, zero);
            qi_[i] = one;
            size_type j = _rank;
            while (0 < j) {
                --j;
                vrow const qrj_ = shadow_matrix_[j]; // containing packed QR
                value_type s_ = zero;
                for (size_type k = j; k < dimension_; ++k) {
                    s_ += qrj_[k] * qi_[k];
                }
                for (size_type k = j; k < dimension_; ++k) {
                    qi_[k] -= qrj_[k] * s_;
                }
            }
        } // matrix_ is Q
    }

    bool
    steal_best(point_list & _basis)
    { // set moves a point which is furthest from affine subspace formed by points of "_basis" set from "outside_" set to "_basis"
        assert(!_basis.empty());
        size_type const rank_ = _basis.size() - 1;
        assert(rank_ < dimension_);
        vrow const origin_ = matrix_[rank_];
        copy_point(_basis.back(), origin_);
        if (!orthonormalize(_basis, rank_, origin_)) {
            return false;
        }
        forward_transformation(rank_);
        vrow const projection_ = shadow_matrix_.back();
        vrow const apex_ = shadow_matrix_.front();
        value_type distance_ = zero; // square of distance to the subspace
        auto const oend = std::cend(outside_);
        auto furthest = oend;
        for (auto it = std::cbegin(outside_); it != oend; ++it) {
            copy_point(*it, apex_);
            subtract_and_assign(projection_, apex_, origin_); // turn translated space into vector space then project onto orthogonal subspace
            for (size_type i = 0; i < rank_; ++i) {
                crow const qi_ = matrix_[i];
                multiply_and_add(projection_, qi_, -std::inner_product(qi_, qi_ + dimension_, apex_, zero));
            }
            value_type d_ = std::inner_product(projection_, projection_ + dimension_, projection_, zero);
            if (distance_ < d_) {
                distance_ = std::move(d_);
                furthest = it;
            }
        }
        if (furthest == oend) {
            return false;
        }
        _basis.splice(std::cend(_basis), outside_, furthest);
        return true;
    }

    facet_array removed_facets_;

    std::pair< facet &, size_type const >
    add_facet(point_array const & _vertices,
              size_type const _against,
              point_iterator const _apex,
              size_type const _neighbour)
    {
        if (removed_facets_.empty()) {
            size_type const f = facets_.size();
            facets_.emplace_back();
            facet & facet_ = facets_.back();
            make_facet(facet_, _vertices, _against, _apex, _neighbour);
            return {facet_, f};
        } else {
            size_type const f = removed_facets_.back();
            removed_facets_.pop_back();
            facet & facet_ = facets_[f];
            reuse_facet(facet_, _vertices, _against, _apex, _neighbour);
            return {facet_, f};
        }
    }

    using ranking = std::multimap< value_type, size_type >;
    using ranking_meta = std::unordered_map< size_type, typename ranking::iterator >;

    ranking ranking_;
    ranking_meta ranking_meta_;

    void
    rank(value_type && _orientation,
         size_type const f)
    {
        if (eps < _orientation) {
            ranking_meta_.emplace(f, ranking_.emplace(std::move(_orientation), f));
        }
    }

    void
    unrank(size_type const f)
    {
        auto const r = ranking_meta_.find(f);
        if (r != std::end(ranking_meta_)) {
            ranking_.erase(r->second);
            ranking_meta_.erase(r);
        }
        removed_facets_.push_back(f);
    }

    point_list outside_;

    value_type
    partition(facet & _facet)
    {
        value_type distance_ = zero;
        auto it = std::cbegin(outside_);
        auto const oend = std::cend(outside_);
        while (it != oend) {
            auto const next = std::next(it);
            value_type d_ = _facet.distance(std::cbegin(**it));
            if (eps < d_) {
                if (distance_ < d_) {
                    distance_ = std::move(d_);
                    _facet.outside_.splice(std::cbegin(_facet.outside_), outside_, it);
                } else {
                    _facet.outside_.splice(std::cend(_facet.outside_), outside_, it);
                }
            } else if (!(d_ < -eps)) {
                _facet.coplanar_.push_back(*it);
            }
            it = next;
        }
        return distance_;
    }

    size_type
    get_best_facet() const
    {
        assert(ranking_meta_.size() == ranking_.size());
        return std::prev(std::cend(ranking_))->second;
    }

    void
    replace_neighbour(size_type const f,
                      size_type const _from,
                      size_type const _to)
    {
        if (_from != _to) {
            for (size_type & n : facets_[f].neighbours_) {
                if (n == _from) {
                    n = _to;
                    return;
                }
            }
        }
    }

    struct ridge
    {

        facet & facet_;
        size_type const f;
        size_type const v;
        size_type const hash_;

        bool
        operator == (ridge const & _rhs) const noexcept
        {
            point_iterator const lskip = facet_.vertices_[v];
            point_iterator const rskip = _rhs.facet_.vertices_[_rhs.v];
            for (point_iterator const & l : facet_.vertices_) {
                if (l != lskip) {
                    bool found_ = false;
                    for (point_iterator const & r : _rhs.facet_.vertices_) {
                        if (r != rskip) {
                            if (l == r) {
                                found_ = true; // O(D^2) expensive
                                break;
                            }
                        }
                    }
                    if (!found_) {
                        return false;
                    }
                }
            }
            return true;
        }

    };

    struct ridge_hash
    {

        size_type
        operator () (ridge const & _ridge) const noexcept
        {
            return _ridge.hash_;
        }

    };

    std::unordered_set< ridge, ridge_hash > unique_ridges_;
    std::hash< typename std::iterator_traits< point_iterator >::value_type const * > point_hash_;
    std::vector< size_type > vertices_hashes_;

    void
    find_adjacent_facets(facet & _facet,
                         size_type const f,
                         size_type const _skip)
    {
        size_type ridge_hash_ = 0;
        for (size_type v = 0; v < dimension_; ++v) {
            if (v != _skip) {
                ridge_hash_ ^= (vertices_hashes_[v] = point_hash_(std::addressof(*_facet.vertices_[v])));
            }
        }
        for (size_type v = 0; v < dimension_; ++v) {
            if (v != _skip) { // neighbouring facet against apex (_skip-indexed) is known atm
                auto const position = unique_ridges_.insert({_facet, f, v, (ridge_hash_ ^ vertices_hashes_[v])});
                if (!position.second) {
                    ridge const & ridge_ = *position.first;
                    ridge_.facet_.neighbours_[ridge_.v] = f;
                    _facet.neighbours_[v] = ridge_.f;
                    unique_ridges_.erase(position.first);
                }
            }
        }
    }

    using facet_unordered_set = std::unordered_set< size_type >;

    facet_unordered_set visited_;
    facet_unordered_set visible_;

    bool
    process_visibles(facet_array & _newfacets,
                     size_type const f,
                     point_iterator const _apex) // traverse the graph of visible facets
    {
        assert(!(visited_.size() < visible_.size()));
        if (!visited_.insert(f).second) {
            return (visible_.count(f) != 0);
        }
        facet & facet_ = facets_[f];
        if (!(zero < facet_.distance(std::cbegin(*_apex)))) {
            return false;
        }
        visible_.insert(f);
        outside_.splice(std::cend(outside_), std::move(facet_.outside_));
        facet_.coplanar_.clear();
        for (size_type v = 0; v < dimension_; ++v) {
            size_type const neighbour = facet_.neighbours_[v];
            if (!process_visibles(_newfacets, neighbour, _apex)) { // recursive function
                auto const newfacet = add_facet(facet_.vertices_, v, _apex, neighbour);
                set_hyperplane_equation(newfacet.first);
                _newfacets.push_back(newfacet.second);
                replace_neighbour(neighbour, f, newfacet.second);
                find_adjacent_facets(newfacet.first, newfacet.second, v);
            }
        }
        unrank(f);
        return true;
    }

    void
    compactify()
    {
        size_type source = facets_.size();
        assert(removed_facets_.size() < source);
        assert(dimension_ < source - removed_facets_.size());
        assert(ranking_.size() == ranking_meta_.size());
        assert(!(source < ranking_.size()));
        auto const rend = std::end(ranking_meta_);
        std::sort(std::rbegin(removed_facets_), std::rend(removed_facets_));
        for (size_type const destination : removed_facets_) {
            assert(!(source < destination));
            if (destination != --source) {
                facet & facet_ = facets_[destination];
                facet_ = std::move(facets_.back());
                for (size_type const n : facet_.neighbours_) {
                    replace_neighbour(n, source, destination);
                }
                auto const r = ranking_meta_.find(source);
                if (r != rend) {
                    r->second->second = destination;
                    ranking_meta_.emplace(destination, std::move(r->second));
                    ranking_meta_.erase(r);
                }
            }
            facets_.pop_back();
        }
        removed_facets_.clear();
    }

    bool
    check_local_convexity(facet const & facet_,
                          size_type const f) const
    {
        assert(&facets_[f] == &facet_);
        for (size_type const n : facet_.neighbours_) {
            facet const & neighbour_ = facets_[n];
            if (cos_of_dihedral_angle(facet_, neighbour_) < one) { // avoid roundoff error
                for (size_type v = 0; v < dimension_; ++v) {
                    if (neighbour_.neighbours_[v] == f) { // vertex v of neigbour_ facet is opposite to facet_
                        value_type const distance_ = facet_.distance(std::cbegin(*neighbour_.vertices_[v]));
                        if (eps < distance_) {
                            return false; // facet is not locally convex at ridge, common for facet_ and neighbour_ facets
                        } else {
                            break;
                        }
                    }
                }
            }
        }
        return true;
    }

public :

    template< typename iterator >
    value_type
    hypervolume(iterator first,
                iterator const last) // hypervolume of parallelotope spanned on vectors from last vertex (vlast) to all the vertices lies in [vfirst, vlast)
    {
        using iterator_traits = std::iterator_traits< iterator >;
        static_assert(std::is_base_of< std::input_iterator_tag, typename iterator_traits::iterator_category >::value);
        static_assert(std::is_constructible< point_iterator, typename iterator_traits::value_type >::value);
        if (first == last) {
            return zero;
        }
        vrow const origin_ = shadow_matrix_.back();
        copy_point(*last, origin_);
        size_type rank_ = 0;
        while (first != last) { // affine space -> vector space
            assert(rank_ < dimension_);
            vrow const row_ = matrix_[rank_];
            copy_point(*first, row_);
            subtract(row_, origin_);
            ++rank_;
            ++first;
        }
        if (rank_ == dimension_) {
            return det(); // oriented hypervolume
        } else {
            matrix_sqr(rank_);
            using std::sqrt;
            return sqrt(det(shadow_matrix_, rank_)); // non-oriented rank_-dimensional measure
        }
    }

    void
    add_points(point_iterator beg,
               point_iterator const end) // [beg; end)
    {
        while (beg != end) {
            outside_.push_back(beg);
            ++beg;
        }
    }

    template< typename iterator >
    void
    add_points(iterator const beg,
               iterator const end) // [beg; end)
    {
        using iterator_traits = std::iterator_traits< iterator >;
        static_assert(std::is_base_of< std::input_iterator_tag, typename iterator_traits::iterator_category >::value);
        static_assert(std::is_constructible< point_iterator, typename iterator_traits::value_type >::value);
        std::copy(beg, end, std::back_inserter(outside_));
    }

    point_list
    get_affine_basis()
    {
        assert(facets_.empty());
        point_list basis_;
        basis_.splice(std::cend(basis_), outside_, std::begin(outside_));
        if (!steal_best(basis_)) {
            return basis_; // can't find affinely independent second point
        }
        outside_.splice(std::cbegin(outside_), basis_, std::begin(basis_)); // reject first point to rejudge it
        for (size_type i = 0; i < dimension_; ++i) {
            if (!steal_best(basis_)) {
                return basis_; // can't find (i + 2) affinely independent points
            }
        }
        return basis_;
    }

    template< typename iterator >
    value_type
    create_initial_simplex(iterator const first,
                           iterator const last) // [bfirst; blast]
    {
        using iterator_traits = std::iterator_traits< iterator >;
        static_assert(std::is_base_of< std::forward_iterator_tag, typename iterator_traits::iterator_category >::value);
        static_assert(std::is_constructible< point_iterator, typename iterator_traits::value_type >::value);
        assert(static_cast< size_type >(std::distance(first, last)) == dimension_);
        assert(facets_.empty());
        {
            copy_point(*last, inner_point_);
            auto it = first;
            while (it != last) {
                auto x = std::cbegin(**it);
                for (size_type i = 0; i < dimension_; ++i) {
                    inner_point_[i] += *x;
                    ++x;
                }
                ++it;
            }
            divide(inner_point_, value_type(dimension_ + 1));
        }
        value_type const volume_ = hypervolume(first, last);
        bool const swap_ = (volume_ < zero);
        for (size_type f = 0; f <= dimension_; ++f) {
            facets_.emplace_back();
            facet & facet_ = facets_.back();
            make_facet(facet_, first, f, swap_);
            set_hyperplane_equation(facet_);
            rank(partition(facet_), f);
        }
        outside_.clear();
        assert(check());
        return volume_;
    }

    // Barber, C. B., D.P. Dobkin, and H.T. Huhdanpaa, 1995. "The Quickhull Algorithm for Convex Hulls", ACM Transactions on Mathematical Software.
    void
    create_convex_hull()
    {
        assert(facets_.size() == dimension_ + 1);
        assert(removed_facets_.empty());
        facet_array newfacets_;
        while (!ranking_.empty()) {
            size_type const f = get_best_facet();
            point_list & o_ = facets_[f].outside_;
            assert(!o_.empty());
            point_iterator const apex = std::move(o_.front());
            o_.pop_front();
            if (!process_visibles(newfacets_, f, apex)) {
                assert(false);
            }
            visited_.clear();
            visible_.clear();
            assert(unique_ridges_.empty());
            for (size_type const n : newfacets_) {
                facet & facet_ = facets_[n];
                assert(check_local_convexity(facet_, n));
                rank(partition(facet_), n);
            }
            newfacets_.clear();
            outside_.clear();
            //assert((compactify(), check()));
        }
        assert(ranking_meta_.empty());
        compactify();
    }

    // Kurt Mehlhorn, Stefan Näher, Thomas Schilz, Stefan Schirra, Michael Seel, Raimund Seidel, and Christian Uhrig.
    // Checking geometric programs or verification of geometric structures. In Proc. 12th Annu. ACM Sympos. Comput. Geom., pages 159–165, 1996.
    bool
    check() const
    {
        assert(dimension_ < facets_.size());
        size_type facets_count_ = 0;
        for (facet const & facet_ : facets_) { // check local convexity of all the facets
            if (!check_local_convexity(facet_, facets_count_)) {
                return false;
            }
            ++facets_count_;
        }
        facet const & first_ = facets_.front();
        {
            value_type const distance_ = first_.distance(inner_point_);
            if (!(distance_ < zero)) {
                return false; // inner point is not on negative side of the first facet, therefore structure is not convex
            }
        }
        vector memory_(dimension_ * (4 + dimension_), zero);
        vrow centroid_ = memory_.data();
        vrow const ray_ = centroid_;
        centroid_ += dimension_;
        for (point_iterator const & v : first_.vertices_) {
            auto x = std::cbegin(*v);
            for (size_type i = 0; i < dimension_; ++i) {
                ray_[i] += *x;
                ++x;
            }
        }
        divide(ray_, value_type(dimension_));
        subtract(ray_, inner_point_);
        {
            value_type const dot_product_ = std::inner_product(ray_, ray_ + dimension_, first_.normal_.data(), zero);
            if (!(zero < dot_product_)) { // ray is parallel to the plane or directed away from the plane
                return false;
            }
        }
        matrix g_{dimension_}; // storage (d * (d + 1)) for Gaussian elimination with partial pivoting
        for (vrow & row_ : g_) {
            row_ = centroid_;
            centroid_ += (dimension_ + 1);
        }
        vrow const intersection_point_ = centroid_;
        centroid_ += dimension_;
        assert(centroid_ + dimension_ == &memory_.back() + 1);
        for (size_type f = 1; f < facets_count_; ++f) {
            using std::abs;
            facet const & facet_ = facets_[f];
            value_type const numerator_ = facet_.distance(inner_point_);
            if (!(numerator_ < zero)) {
                return false; // inner point is not on negative side of all the facets, i.e. structure is not convex
            }
            value_type const denominator_ = std::inner_product(ray_, ray_ + dimension_, facet_.normal_.data(), zero);
            if (!(zero < denominator_)) { // ray is parallel to the plane or directed away from the plane
                continue;
            }
            std::copy_n(ray_, dimension_, intersection_point_);
            scale_and_shift(intersection_point_, inner_point_, -(numerator_ / denominator_));
            for (size_type v = 0; v < dimension_; ++v) {
                auto beg = std::cbegin(*facet_.vertices_[v]);
                for (size_type r = 0; r < dimension_; ++r) {
                    g_[r][v] = *beg;
                    ++beg;
                }
            }
            for (size_type r = 0; r < dimension_; ++r) {
                vrow const gr_ = g_[r];
                centroid_[r] = -std::accumulate(gr_, gr_ + dimension_, zero) / value_type(dimension_);
                gr_[dimension_] = intersection_point_[r];
            }
            for (size_type r = 0; r < dimension_; ++r) {
                vrow const gr_ = g_[r];
                value_type & x_ = centroid_[r];
                gshift(gr_, x_);
                //assert(!(eps * value_type(dimension_) < std::accumulate(gr_, gr_ + dimension_, zero))); // now center of the facet coincides with the origin, but no one vertex does
                auto const bounding_box = std::minmax_element(gr_, gr_ + dimension_);
                x_ = *bounding_box.second - *bounding_box.first;
                if (!(eps * value_type(dimension_) < x_)) {
                    x_ = one;
                }
            }
            for (size_type r = 0; r < dimension_; ++r) {
                vrow const gr_ = g_[r];
                gshift(gr_, centroid_[r]);
            }
            for (size_type i = 0; i < dimension_; ++i) { // Gaussian elimination
                vrow & gi_ = g_[i];
                value_type max_ = abs(gi_[i]);
                size_type pivot = i;
                {
                    size_type p = i;
                    while (++p < dimension_) {
                        value_type y_ = abs(g_[p][i]);
                        if (max_ < y_) {
                            max_ = std::move(y_);
                            pivot = p;
                        }
                    }
                }
                assert(eps < max_); // vertex must not match the origin after above transformations
                if (pivot != i) {
                    std::swap(gi_, g_[pivot]);
                }
                value_type & gii_ = gi_[i];
                for (size_type j = i + 1; j < dimension_; ++j) {
                    vrow const gj_ = g_[j];
                    value_type & gji_ = gj_[i];
                    gji_ /= gii_;
                    for (size_type k = i + 1; k <= dimension_; ++k) {
                        gj_[k] -= gji_ * gi_[k];
                    }
                    gji_ = zero;
                }
            } // g_ is upper triangular now
            bool in_range_ = true;
            {
                size_type i = dimension_;
                while (0 < i) {
                    --i;
                    vrow const gi_ = g_[i];
                    value_type & xi_ = gi_[dimension_];
                    for (size_type j = i + 1; j < dimension_; ++j) {
                        xi_ -= gi_[j] * g_[j][dimension_];
                    }
                    value_type const & gii_ = gi_[i];
                    assert(eps < abs(gii_)); // vertex must not match the origin
                    xi_ /= gii_;
                    if ((xi_ < zero) || (one < xi_)) {
                        in_range_ = false; // barycentric coordinate does not lie in [0;1] interval => miss
                        break;
                    }
                }
            }
            if (in_range_) {
                return false; // hit
            }
        }
        return true;
    }

};