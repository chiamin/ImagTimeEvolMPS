import Random

function mix_ind(xi, yi, Lx, Ly, first_index=1, first_x_index=1)
    return (xi-first_x_index) * Ly + (yi-first_x_index) + first_index
end

function Add_hopping(ampo, i, j, t)
    if t != 0
        ampo += -t,"Cdagup",i,"Cup",j
        ampo += -conj(t),"Cdagup",j,"Cup",i
        ampo += -t,"Cdagdn",i,"Cdn",j
        ampo += -conj(t),"Cdagdn",j,"Cdn",i
    end
    return ampo
end

function Hubbard(Lx, Ly, tx, ty, U, tpr, tppx, tppy, mu, periodic_x, periodic_y)
    ampo = OpSum()

    N = Lx * Ly
    for j=1:N
        if U != 0
            ampo += U,"Nupdn",j
        end
        if mu != 0
            ampo += -mu,"Ntot",j
        end
    end

    for xi=1:Lx
        for yi=1:Ly
            i = mix_ind(xi,yi,Lx,Ly)
            xp = xi+1
            yp = yi+1
            xn = xi-1
            xpp = xi+2
            ypp = yi+2

            if xp > Lx && periodic_x
                xp -= Lx
            end
            if yp > Ly && periodic_y
                yp -= Ly
            end
            if xn < 1 && periodic_x
                xn += Lx
            end
            if xpp > Lx && periodic_x
                xpp -= Lx
            end
            if ypp > Ly && periodic_y
                ypp -= Ly
            end


            if xp <= Lx
                j = mix_ind(xp,yi,Lx,Ly)
                ampo = Add_hopping(ampo, i, j, tx)
            end

            if yp <= Ly
                j = mix_ind(xi,yp,Lx,Ly)
                ampo = Add_hopping(ampo, i, j, ty)

                if xp <= Lx
                    j = mix_ind(xp,yp,Lx,Ly)
                    ampo = Add_hopping(ampo, i, j, tpr)
                end

                if xn >= 1
                    j = mix_ind(xn,yp,Lx,Ly)
                    ampo = Add_hopping(ampo, i, j, tpr)
                end
            end

            if xpp <= Lx
                if !(xpp < xi && Lx <= 4)
                    j = mix_ind(xpp,yi,Lx,Ly)
                    ampo = Add_hopping(ampo, i, j, tppx)
                end
            end
            if ypp <= Ly
                if !(ypp < yi && Ly <= 4)
                  j = mix_ind(xi,ypp,Lx,Ly)
                  ampo = Add_hopping(ampo, i, j, tppy)
                end
            end
        end
    end
    return ampo
end

function Add_hopping_spinless(ampo, i, j, t)
    if t != 0
        ampo += -t,"Cdag",i,"C",j
        ampo += -conj(t),"Cdag",j,"C",i
    end
    return ampo
end

function SpinlessHamilt(Lx, Ly, tx, ty, tpr, tppx, tppy, mu, periodic_x, periodic_y)
    ampo = OpSum()

    N = Lx * Ly
    if mu != 0
        for j=1:N
            ampo += -mu,"N",j
        end
    end

    for xi=1:Lx
        for yi=1:Ly
            i = mix_ind(xi,yi,Lx,Ly)
            xp = xi+1
            yp = yi+1
            xn = xi-1
            xpp = xi+2
            ypp = yi+2

            if xp > Lx && periodic_x
                xp -= Lx
            end
            if yp > Ly && periodic_y
                yp -= Ly
            end
            if xn < 1 && periodic_x
                xn += Lx
            end
            if xpp > Lx && periodic_x
                xpp -= Lx
            end
            if ypp > Ly && periodic_y
                ypp -= Ly
            end


            if xp <= Lx
                j = mix_ind(xp,yi,Lx,Ly)
                ampo = Add_hopping_spinless(ampo, i, j, tx)
            end

            if yp <= Ly
                j = mix_ind(xi,yp,Lx,Ly)
                ampo = Add_hopping_spinless(ampo, i, j, ty)

                if xp <= Lx
                    j = mix_ind(xp,yp,Lx,Ly)
                    ampo = Add_hopping_spinless(ampo, i, j, tpr)
                end

                if xn >= 1
                    j = mix_ind(xn,yp,Lx,Ly)
                    ampo = Add_hopping_spinless(ampo, i, j, tpr)
                end
            end

            if xpp <= Lx
                if !(xpp < xi && Lx <= 4)
                    j = mix_ind(xpp,yi,Lx,Ly)
                    ampo = Add_hopping_spinless(ampo, i, j, tppx)
                end
            end
            if ypp <= Ly
                if !(ypp < yi && Ly <= 4)
                  j = mix_ind(xi,ypp,Lx,Ly)
                  ampo = Add_hopping_spinless(ampo, i, j, tppy)
                end
            end
        end
    end
    return ampo
end

function Hk_onebody(Lx, Ly, tx, ty, tpr, tppx, tppy, periodic_x, periodic_y)
    N = Lx * Ly

    Hk = zeros(N,N)
    for xi=1:Lx
        for yi=1:Ly
            i = mix_ind(xi,yi,Lx,Ly)
            xp = xi+1
            yp = yi+1
            xn = xi-1
            xpp = xi+2
            ypp = yi+2

            if xp > Lx && periodic_x
                xp -= Lx
            end
            if yp > Ly && periodic_y
                yp -= Ly
            end
            if xn < 1 && periodic_x
                xn += Lx
            end
            if xpp > Lx && periodic_x
                xpp -= Lx
            end
            if ypp > Ly && periodic_y
                ypp -= Ly
            end


            if xp <= Lx
                j = mix_ind(xp,yi,Lx,Ly)
                Hk[i,j] = Hk[j,i] = -tx
            end

            if yp <= Ly
                j = mix_ind(xi,yp,Lx,Ly)
                Hk[i,j] = Hk[j,i] = -ty

                if xp <= Lx
                    j = mix_ind(xp,yp,Lx,Ly)
                    Hk[i,j] = Hk[j,i] = -tpr
                end

                if xn >= 1
                    j = mix_ind(xn,yp,Lx,Ly)
                    Hk[i,j] = Hk[j,i] = -tpr
                end
            end

            if xpp <= Lx
                if !(xpp < xi && Lx <= 4)
                    j = mix_ind(xpp,yi,Lx,Ly)
                    Hk[i,j] = Hk[j,i] = -tppx
                end
            end
            if ypp <= Ly
                if !(ypp < yi && Ly <= 4)
                  j = mix_ind(xi,ypp,Lx,Ly)
                  Hk[i,j] = Hk[j,i] = -tppy
                end
            end
        end
    end
    return Hk
end

function RandomState(Nsite; Nup, Ndn, seed=0)
    if seed != 0
        println("RandomMPS seed =",seed)
        Random.seed!(seed)
    end

    notfull = zeros(Int, Nsite)
    for i=1:Nsite
        notfull[i] = i
    end

    states = fill("Emp",Nsite)

    while Nup > 0 || Ndn > 0
        j = rand(1:length(notfull))
        site = notfull[j]
        state = states[site]
        if state == "Emp"
            if Ndn <= 0
                state = "Up"
                Nup -= 1
            elseif Nup <= 0
                state = "Dn"
                Ndn -= 1
            else
                choose_up = rand((true,false))
                if choose_up
                    state = "Up"
                    Nup -= 1
                else
                    state = "Dn"
                    Ndn -= 1
                end
            end
        elseif state == "Up"
            if Ndn > 0
                state = "UpDn"
                Ndn -= 1
            end
            deleteat!(notfull, j)
        elseif state == "Dn"
            if Nup > 0
                state = "UpDn"
                Nup -= 1
            end
            deleteat!(notfull, j)
        end
        states[site] = state
    end

    println("Random state:",states)
    return states
end

function RandomConf(Nsites; Nup, Ndn, seed=0)
    @assert Nup <= Nsites && Ndn <= Nsites
    state = RandomState(Nsites; Nup=Nup, Ndn=Ndn, seed=seed)
    conf = Vector{Int}()
    for s in state
        if s == "Emp"
            push!(conf, 1)
        elseif s == "Up"
            push!(conf, 2)
        elseif s == "Dn"
            push!(conf, 3)
        elseif s == "UpDn"
            push!(conf, 4)
        else
            error("Unknow state")
        end
    end
    return conf
end
