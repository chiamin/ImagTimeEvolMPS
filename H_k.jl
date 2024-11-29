
# Generate the one-body kinetic term of the Hubbard Hamiltonian with the given parameters
# Input:
#   Lx: The number of lattice sites in the x direction.
#   Ly: The number of lattice sites in the y direction.
#   Lz: The number of lattice sites in the z direction.
#   tx: The hopping amplitude between nearest-neighbor sites in the x direction
#   ty: The hopping amplitude between nearest neighbor sites in the y direction
#   tz: The hopping amplitude between nearest neighbor sites in the y direction
# Output
#   H: The one-body kinetic Hamiltonian in the form of a square matrix of size (Lx*Ly*Lz) 
#function H_K(Lx::Int64, Ly::Int64, Lz::Int64, tx::Float64, ty::Float64, tz::Float64, xpbc::Bool, ypbc::Bool, zpbc::Bool)::Matrix{Float64}
function H_K(Lx::Int64, Ly::Int64, Lz::Int64, tx::Float64, ty::Float64, tz::Float64, xpbc::Bool, ypbc::Bool, zpbc::Bool)::Matrix{Float64}
    r = 0
    N_sites = Lx*Ly*Lz
    H = zeros(N_sites,N_sites)

    for mz=1:Lz
        for iy=1:Ly
            for jx=1:Lx
                r=r+1      # r=(iy-1)*Lx+jx
                if Lx!=1
                    if jx==1
                        if xpbc
                            H[r,r+Lx-1] = -tx
                        end
                        H[r,r+1] = -tx
                    elseif jx==Lx
                        H[r,r-1] = -tx
                        if xpbc
                            H[r,r+1-Lx] = -tx
                        end
                    else
                        H[r,r-1] = -tx
                        H[r,r+1] = -tx
                    end
                end

                if Ly!=1
                    if iy==1
                        if ypbc
                            H[r,r+(Ly-1)*Lx] = -ty
                        end
                        H[r,r+Lx] = -ty
                    elseif iy==Ly
                        H[r,r-Lx] = -ty
                        if ypbc
                            H[r,r-(Ly-1)*Lx] = -ty
                        end
                    else
                        H[r,r-Lx] = -ty
                        H[r,r+Lx] = -ty
                    end
                end

                if Lz!=1
                    if mz==1
                        if zpbc
                            H[r,r+(Lz-1)*Lx*Ly] = -tz
                        end
                        H[r,r+Lx*Ly] = -tz
                    elseif mz==Lz
                        H[r,r-Lx*Ly] = -tz
                        if zpbc
                            H[r,r-(Lz-1)*Lx*Ly] = -tz
                        end
                    else
                        H[r,r-Lx*Ly] = -tz
                        H[r,r+Lx*Ly] = -tz
                    end
                end
            end
        end
    end
    return H
end
