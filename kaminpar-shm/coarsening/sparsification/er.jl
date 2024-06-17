using Laplacians
using SparseArrays
using LinearAlgebra

struct C_IJV
    i::Array{Int64,1}
    j::Array{Int64,1}
    v::Array{Cdouble,1}
end

# based on sparsify method in module Lapacians
function effective_resistances(g::C_IJV)::C_IJV
  a = sparse(g.i, g.j, g.v)

  f = approxchol_lap(a,tol = 1e-2);

  n = size(a,1)
  JLfac = 0.4
  k = round(Int, JLfac*log(n)) # number of dims for Johnson-Lindenstrauss

  U = wtedEdgeVertexMat(a)
  m = size(U,1)
  R = randn(Float64, m,k)
  UR = U'*R;

  V = zeros(n,k)
  for i in 1:k
    V[:, i] = f(UR[:, i])
  end

  (ai,aj,av) = findnz(triu(a))
  ers = zeros(size(av))
  for h in 1:length(av)
    i = ai[h]
    j = aj[h]
    # divided by k, because 1/sqrt(k)
    ers[h] = min(1,av[h] * ((norm(V[i, :] - V[j, :])^2) / k))
    #        min(1, w_e * (R_e                      ))
  end
  erMatrix = sparse(ai,aj,ers, n, n )
  erMatrix = erMatrix + erMatrix'

  (i,j,v) = findnz(erMatrix)
  return C_IJV(i,j,v)
  return
end

(i,j,v) = findnz(sparse([
0 2 3 1;
2 0 1 0;
3 1 0 0;
1 0 0 0
]))
print(effective_resistances(C_IJV(i,j,v)))