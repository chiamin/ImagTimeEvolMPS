
mutable struct SquareLattice
    Lx::Int
    Ly::Int
    xPBC::Bool
    yPBC::Bool
    nb::Vector{Vector{Int}}
    bonds::Vector{Vector{Int}}
end

function coordToIndex(x::Int, y::Int, Ly::Int)::Int
    return (x-1)*Ly + y
end

function makeSquareLattice(Lx::Int, Ly::Int, xpbc::Bool, ypbc::Bool)
    latt = SquareLattice(Lx, Ly, xpbc, ypbc, [Vector{Int}() for i in 1:Lx*Ly], [])

    function add_nb!(latt, i1, i2)
        push!(latt.nb[i1], i2)
        if i1 < i2 && !([i1,i2] in latt.bonds)
            push!(latt.bonds, [i1,i2])
        end
    end

    for x=1:Lx
        for y=1:Ly
            i = coordToIndex(x, y, Ly)

            xp = x+1
            if xpbc && xp > Lx
                xp -= Lx
            end
            if 1 <= xp <= Lx
                inb = coordToIndex(xp, y, Ly)
                add_nb!(latt, i, inb)
            end

            xn = x-1
            if xpbc && xn < 1
                xn += Lx
            end
            if 1 <= xn <= Lx
                inb = coordToIndex(xn, y, Ly)
                add_nb!(latt, i, inb)
            end

            yp = y+1
            if ypbc && yp > Ly
                yp -= Ly
            end
            if 1 <= yp <= Ly
                inb = coordToIndex(x, yp, Ly)
                add_nb!(latt, i, inb)
            end

            yn = y-1
            if ypbc && yn < 1
                yn += Ly
            end
            if 1 <= yn <= Ly
                inb = coordToIndex(x, yn, Ly)
                add_nb!(latt, i, inb)
            end
        end
    end
    return latt
end
