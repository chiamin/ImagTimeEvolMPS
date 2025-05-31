using Test
include("../SquareLattice.jl")

function SquareLattice_Test()
    Lx, Ly = 4, 1
    latt = makeSquareLattice(Lx,Ly,false,false)
    for bond in latt.bonds
        i,j = bond
        @test 1 <= i <= Lx*Ly
        @test 1 <= j <= Lx*Ly
    end

    Lx = Ly = 3
    latt = makeSquareLattice(Lx,Ly,true,true)
    # 3 6 9
    # 2 5 8
    # 1 4 7
    @test latt.nb[1] == [4,7,2,3]

    @test length(latt.bonds) == 18
    @test Set(latt.bonds) == Set([[1,4],[1,7],[1,2],[1,3],[2,5],[2,8],[2,3],[3,6],[3,9],[4,7],[4,5],[4,6],[5,8],[5,6],[6,9],[7,8],[7,9],[8,9]])

    latt = makeSquareLattice(Lx,Ly,false,false)
    # 3 6 9
    # 2 5 8
    # 1 4 7
    @test latt.nb[1] == [4,2]

    @test length(latt.bonds) == 12
    @test Set(latt.bonds) == Set([[1,4],[1,2],[2,5],[2,3],[3,6],[4,7],[4,5],[5,8],[5,6],[6,9],[7,8],[8,9]])
end

SquareLattice_Test()
