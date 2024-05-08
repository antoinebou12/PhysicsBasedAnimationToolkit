from .. import codegen as cg
import sympy as sp
from sympy.polys.monomials import itermonomials
from sympy.polys.orderings import monomial_key
import numpy as np
import argparse


def dot(u, v, x: list, a: list, b: list):
    assert(len(a) == len(b) == len(x))
    I = u*v
    for i in reversed(range(len(a))):
        I = sp.integrate(I, (x[i], a[i], b[i])).expand()
    return I


def norm(u, x: list, a: list, b: list):
    return sp.sqrt(dot(u, u, x, a, b))


def proj(u, v, x: list, a: list, b: list):
    return dot(u, v, x, a, b) / dot(v, v, x, a, b) * v


def gram_schmidt(V: list, x: list, a: list, b: list):
    U = []
    for vi in V:
        ui = vi
        for uj in U:
            ui = ui - proj(vi, uj, x, a, b)
        U.append(ui)
    return U


def normalize(V: list, x: list, a: list, b: list):
    U = []
    for vi in V:
        ui = vi / norm(vi, x, a, b)
        U.append(ui)
    return U


def orthonormalized_basis(dims, order, x, a, b):
    V = []
    if dims == 1:
        # Generate the monomial basis
        Vl = sorted(itermonomials(x, order), key=monomial_key('grlex', list(reversed(x))))
        # Orthonormalize the line's polynomial basis
        Vlo = gram_schmidt(Vl, x, a, b)
        V = normalize(Vlo, x, a, b)
    if dims == 2:
        # Generate the monomial basis
        Vf = sorted(itermonomials(x, order), key=monomial_key('grlex', list(reversed(x))))
        # Orthonormalize the triangle's polynomial basis'
        Vfo = gram_schmidt(Vf, x, a, b)
        V = normalize(Vfo, x, a, b)
    if dims == 3:
        # Generate the monomial basis
        Vt = sorted(itermonomials(x, order), key=monomial_key('grlex', list(reversed(x))))
        # Orthonormalize the tetrahedron's polynomial basis'
        Vto = gram_schmidt(Vt, x, a, b)
        V = normalize(Vto, x, a, b)
    for i in range(len(V)):
        V[i] = sp.expand(V[i])
        V[i] = sp.factor(V[i])
    return V


def vector_valued_basis(V: list, dims: int):
    G = []
    for vi in V:
        for d in range(dims):
            gn = [0]*dims
            gn[d] = vi
            G.append(gn)
    return G


def divergence(u: list, x: list):
    div = 0
    for i in range(len(x)):
        div = div + sp.diff(u[i], x[i])
    return div


def divergence_free_basis(V: list, dims: int, x: list, a: list, b: list):
    G = vector_valued_basis(V, dims)
    H = sp.Matrix.zeros(len(V), dims*len(V))
    for n in range(len(G)):
        gn = G[n]
        assert(len(gn) == len(x))
        div = divergence(gn, x)
        # Compute the coefficients of div(gn) w.r.t. orthonormal basis V
        for i in range(len(V)):
            H[i,n] = dot(div, V[i], x, a, b)

    HN = H.nullspace()
    GG = sp.Matrix(G).transpose()
    F = []
    for i in range(len(HN)):
        fi = GG * HN[i]
        F.append(sp.simplify(fi))
    for fi in F:
        div = sp.simplify(divergence(fi, x))
        assert(div == 0)
    return F

def header(file):
    file.write(
"""#ifndef PBA_CORE_MATH_POLYNOMIAL_BASIS_H
#define PBA_CORE_MATH_POLYNOMIAL_BASIS_H

/**
* @file PolynomialBasis.h
*
* All the polynomials defined are based on expressions computed symbolically in the script
* polynomial_basis.py (or equivalently polynomial_basis.ipynb).
*
*/

#include "pba/aliases.h"

#include <cmath>
#include <numbers>

namespace pba {
namespace math {

template <int Dims, int Order>
class MonomialBasis;

template <int Dims, int Order>
class OrthonormalPolynomialBasis;

template <int Dims, int Order>
class DivergenceFreePolynomialBasis;
"""
    )

def footer(file):
    file.write(
"""
} // namespace math
} // namespace pba

#endif // PBA_CORE_MATH_POLYNOMIAL_BASIS_H
"""
    )
    

def codegen(file, V: list, X: list, order: int, orthonormal=False):
    dimV = len(V)
    nvariables = len(X)
    V = sp.Matrix(V)

    classname = "OrthonormalPolynomialBasis" if orthonormal else "MonomialBasis"
    file.write(
"""
template <>
class {0}<{1}, {2}>
{{
  public:
    inline static constexpr std::size_t Dims = {1};
    inline static constexpr std::size_t Order = {2};
    inline static constexpr std::size_t Size = {3};

    [[maybe_unused]] Vector<Size> eval([[maybe_unused]] Vector<{1}> const& X) const 
    {{
        Vector<Size> P;
""".format(classname, nvariables, order, dimV))
    
    code = cg.codegen(V, lhs=sp.MatrixSymbol("P", len(V), 1))
    file.write(cg.tabulate(code, spaces=8))
    
    file.write(
"""
        return P;
    }""")

    file.write("""
               
    [[maybe_unused]] Matrix<Dims, Size> derivatives([[maybe_unused]] Vector<{0}> const& X) const
    {{
        Matrix<Dims, Size> Gm;
        Scalar* G = Gm.data();
""".format(nvariables))

    GV = V.jacobian(X)
    code = cg.codegen(GV, lhs=sp.MatrixSymbol("G", *GV.shape))
    file.write(cg.tabulate(code, spaces=8))

    file.write(
"""
        return Gm;
    }""")

    file.write("""
               
    [[maybe_unused]] Matrix<Size, Dims> antiderivatives([[maybe_unused]] Vector<{0}> const& X) const
    {{
        Matrix<Size, Dims> Pm;
        Scalar* P = Pm.data();
""".format(nvariables))
    
    AV = sp.Matrix([[sp.integrate(V[i], X[d]) for i in range(len(V))] for d in range(len(X))])
    code = cg.codegen(AV, lhs=sp.MatrixSymbol("P", *AV.shape))
    file.write(cg.tabulate(code, spaces=8))

    file.write(
"""
        return Pm;
    }""")

    file.write("""
};
""")


def div_free_codegen(file, F: list, X: list, order: int):
    nvariables = len(X)
    dimF = len(F)
    classname = "DivergenceFreePolynomialBasis"
    file.write(
"""
template <>
class {0}<{1}, {2}>
{{
  public:
    inline static constexpr std::size_t Dims = {1};
    inline static constexpr std::size_t Order = {2};
    inline static constexpr std::size_t Size = {3};

    [[maybe_unused]] Matrix<Size, Dims> eval([[maybe_unused]] Vector<{1}> const& X) const 
    {{
        Matrix<Size, Dims> Pm;
        Scalar* P = Pm.data();
""".format(classname, nvariables, order, dimF))
    
    F = sp.Matrix(F).transpose()
    code = cg.codegen(F, lhs=sp.MatrixSymbol("P", *F.shape))
    file.write(cg.tabulate(code, spaces=8))

    file.write(
"""
        return Pm;
    }
};
""")

if __name__ == "__main__":
    X = sp.Matrix(sp.MatrixSymbol("X", 3, 1))

    parser = argparse.ArgumentParser(
        prog="Polynomial Basis Computer",
        description="""Computes polynomial basis on reference simplices (line, triangle, tetrahedron) and generates C++ code for their computation.""",
    )
    parser.add_argument("-o", "--order", help="Maximum degree of the polynomial basis functions", type=int,
                        dest="order", default=1)
    parser.add_argument("-d", "--dry-run", help="Prints computed polynomial basis' to stdout", type=bool, dest="dry_run")
    args = parser.parse_args()

    # polynomial order
    order = args.order
    dry_run = args.dry_run

    # Bounds and basis for line integral
    la = [0]
    lb = [1]
    xl = [X[0]]
    # Bounds for triangle integral
    fa = [0, 0]
    fb = [1, 1-X[0]]
    xf = [X[0], X[1]]
    # Bounds for tetrahedron integral
    ta = [0, 0, 0]
    tb = [1, 1-X[0], 1-X[0]-X[1]]
    xt = [X[0], X[1], X[2]]

    if dry_run:
        Vlo = orthonormalized_basis(1, order, xl, la, lb)
        print("Orthonormal Polynomial Basis of order={} on the reference line segment".format(order))
        for v in Vlo:
            print(v)

        Vfo = orthonormalized_basis(2, order, xf, fa, fb)
        print("Orthonormal Polynomial Basis of order={} on the reference triangle".format(order))
        for v in Vfo:
            print(v)

        Vto = orthonormalized_basis(2, order, xt, ta, tb)
        print("Orthonormal Polynomial Basis of order={} on the reference tetrahedron".format(order))
        for v in Vto:
            print(v)
    else:
        with open('PolynomialBasis.h', 'w', encoding="utf-8") as file:
            header(file)

            file.write("\n/**\n * Monomial basis in 1D\n */\n")
            for o in range(order):
                V = sorted(itermonomials(xl, o+1), key=monomial_key('lex', list(reversed(xl))))
                codegen(file, V, xl, o+1)
            file.write("\n/**\n * Monomial basis in 2D\n */\n")
            for o in range(order):
                V = sorted(itermonomials(xf, o+1), key=monomial_key('lex', list(reversed(xf))))
                codegen(file, V, xf, o+1)
            file.write("\n/**\n * Monomial basis in 3D\n */\n")
            for o in range(order):
                V = sorted(itermonomials(xt, o+1), key=monomial_key('lex', list(reversed(xt))))
                codegen(file, V, xt, o+1)

            Vl = []
            dVl = []
            for o in range(order):
                V = orthonormalized_basis(1,o+1, xl, la, lb)
                Vl.append(V)
                dVl.append(divergence_free_basis(V, 1, xl, la, lb))

            Vf = []
            dVf = []
            for o in range(order):
                V = orthonormalized_basis(2,o+1, xf, fa, fb)
                Vf.append(V)
                dVf.append(divergence_free_basis(V, 2, xf, fa, fb))

            Vt = []
            dVt = []
            for o in range(order):
                V = orthonormalized_basis(3,o+1, xt, ta, tb)
                Vt.append(V)
                dVt.append(divergence_free_basis(V, 3, xt, ta, tb))

            file.write("\n/**\n * Orthonormalized polynomial basis on reference line\n */\n")
            for o in range(order):
                codegen(file, Vl[o], xl, o+1, orthonormal=True)
            
            file.write("\n/**\n * Orthonormalized polynomial basis on reference triangle\n */\n")
            for o in range(order):
                codegen(file, Vf[o], xf, o+1, orthonormal=True)

            file.write("\n/**\n * Orthonormalized polynomial basis on reference tetrahedron\n */\n")
            for o in range(order):
                codegen(file, Vt[o], xt, o+1, orthonormal=True)

            file.write("\n/**\n * Divergence free polynomial basis on reference line\n */\n")
            for o in range(order):
                div_free_codegen(file, dVl[o], xl, o+1)

            file.write("\n/**\n * Divergence free polynomial basis on reference triangle\n */\n")
            for o in range(order):
                div_free_codegen(file, dVf[o], xf, o+1)

            file.write("\n/**\n * Divergence free polynomial basis on reference tetrahedron\n */\n")
            for o in range(order):
                div_free_codegen(file, dVt[o], xt, o+1)

            footer(file)
