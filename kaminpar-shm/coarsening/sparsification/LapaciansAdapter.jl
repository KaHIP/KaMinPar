"""
# module LapaciansAdapter
"""
module LapaciansAdapter
    using Laplacians
    using SparseArrays

    struct C_IJV
        i::Array{Cint,1}
        j::Array{Cint,1}
        v::Array{Cdouble,1}
        n::Cint
    end


    function sparsify_adapter(A::C_IJV, eps::Cfloat)::C_IJV
        sparsifyed = sparsify(sparse(A.i, A.j, A.v),ep = eps)
        (i, j, v) = findnz(sparsifyed)
        return C_IJV(i, j, v, A.n)
    end

    #=
     0 2 3 1
     2 0 1 0
     3 1 0 0
     1 0 0 0
    =#
    # A = C_IJV([1, 1, 1, 2, 2, 3, 3, 4], [2, 3, 4, 1, 3, 2, 1, 1], [2, 3, 1, 2, 1, 3, 1, 1], 4)
    X = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    (i,j,v) = findnz(sparse(fill(30.0, (20,20))) - sparse(X, X,fill(30,20)))
    sparsfied = sparsify_adapter(C_IJV(i,j,v,5),Cfloat(2))
    print(sparsfied)
    # print("sparsfied: ", sparsfied.i, " ", sparsfied.j, " ", sparsfied.v, " ", sparsfied.n, "\n")
    SM = sparse(sparsfied.i, sparsfied.j, sparsfied.v)
    M = Matrix(SM)
    for row in 1:size(M,1)
        print(M[row, :], "\n")
    end
    print(nnz(SM)/(20*20), "\n")
end
