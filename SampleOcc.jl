function getParNums(conf::Vector{Int})::Tuple{Int,Int}
    Nup, Ndn = 0, 0
    for i in conf
        if i == 1
            nup = 0
            ndn = 0
        elseif i == 2
            nup = 1
            ndn = 0
        elseif i == 3
            nup = 0
            ndn = 1
        elseif i == 4
            nup = 1
            ndn = 1
        else
            error("Invalid state:"*string(i))
        end
        Nup += nup
        Ndn += ndn
    end
    return Nup, Ndn
end

function sampleOccs(latt::Latt, occs::Vector{Int})::Tuple{Vector{Int},Bool} where Latt
    occs = copy(occs)
    ind = rand(1:length(occs))          # randomly choose one of the occupied site
    site = occs[ind]
    nbs::Vector{Int} = latt.nb[site]             # get the neighbors of this site
    site2 = rand(nbs)                   # randomly choose one of the neighboring sites
    if site2 in occs                    # Already exist, return the original occupation
        return occs, false
    else                                # Move to a new site
        occs[ind] = site2
        return occs, true
    end
end


function sampleOccs(Nsites::Int, occs::Vector{Int})::Tuple{Vector{Int},Bool}
    occs = copy(occs)
    ind = rand(1:length(occs))          # randomly choose one of the occupied site
    site = occs[ind]
    site2 = rand(1:Nsites)              # randomly choose one of the site
    if site2 in occs                    # Already exist, return the original occupation
        return occs, false
    else                                # Move to a new site
        occs[ind] = site2
        return occs, true
    end
end

# Additional configurations for two sites
# 1: Emp, 2: Up, 3: Dn, 4: UpDn
# 4: 0  0  ->                   1+1=2, 2+2=4, 3+3=6, 4+4=8
# 4: 0  up -> up 0              1+2=2+1=3, 1+3=3+1=4
# 4: 2  up -> up 2              4+2=2+4=6, 4+3=3+4=7
# 4: up dn -> 2 0, 0 2, dn up   2+3=3+2=5, 4+1=1+4=5
function twoSiteNewConfs_ConservePar(c1::Int, c2::Int)::Vector{Vector{Int}}
    @assert 1 <= c1 <= 4
    @assert 1 <= c2 <= 4
    csum = c1+c2
    if c1 == c2
        return []
    elseif csum == 3 || csum == 4 || csum == 6 || csum == 7
        return [[c2,c1]]
    else
        if c1 == 2 || c2 == 2
            return [[c2,c1], [1,4], [4,1]]
        else
            return [[c2,c1], [2,3], [3,2]]
        end
    end
end

# Additional configurations for two sites
# 1: Emp, 2: Up, 3: Dn, 4: UpDn
# 4: 0  0  ->                   1+1=2, 2+2=4, 3+3=6, 4+4=8
# 4: 0  up -> up 0              1+2=2+1=3, 1+3=3+1=4
# 4: 2  up -> up 2              4+2=2+4=6, 4+3=3+4=7
# 4: up dn -> 2 0, 0 2, dn up   2+3=3+2=5, 4+1=1+4=5
function AllConfs(c1::Int, c2::Int)::Vector{Vector{Int}}
    @assert 1 <= c1 <= 4
    @assert 1 <= c2 <= 4
    csum = c1+c2
    if c1 == c2
        return [[c1,c2]]
    elseif csum == 3 || csum == 4 || csum == 6 || csum == 7
        return [[c1,c2],[c2,c1]]
    else
        if c1 == 2 || c2 == 2
            return [[c1,c2], [c2,c1], [1,4], [4,1]]
        else
            return [[c1,c2], [c2,c1], [2,3], [3,2]]
        end
    end
end

function twoSiteConfs_ConservePar_hard(c1::Int, c2::Int)::Vector{Vector{Int}}
    if c1==1 && c2==1
        return []
    elseif c1==1 && c2==2
        return [[2, 1]]
    elseif c1==1 && c2==3
        return [[3, 1]]
    elseif c1==1 && c2==4
        return [[4, 1], [2, 3], [3, 2]]
    elseif c1==2 && c2==1
        return [[1, 2]]
    elseif c1==2 && c2==2
        return []
    elseif c1==2 && c2==3
        return [[3, 2], [1, 4], [4, 1]]
    elseif c1==2 && c2==4
        return [[4, 2]]
    elseif c1==3 && c2==1
        return [[1, 3]]
    elseif c1==3 && c2==2
        return [[2, 3], [1, 4], [4, 1]]
    elseif c1==3 && c2==3
        return []
    elseif c1==3 && c2==4
        return [[4, 3]]
    elseif c1==4 && c2==1
        return [[1, 4], [2, 3], [3, 2]]
    elseif c1==4 && c2==2
        return [[2, 4]]
    elseif c1==4 && c2==3
        return [[3, 4]]
    elseif c1==4 && c2==4
        return []
    end
end
