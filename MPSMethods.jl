using ITensors, ITensorMPS
N = 10
chi = 4
sites = siteinds("S=1/2",N)
psi = random_mps(sites;linkdims=chi)
magz = expect(psi,"Sz")
for (j,mz) in enumerate(magz)
    println("$j $mz")
end
