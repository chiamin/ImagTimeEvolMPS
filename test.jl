using PyPlot

# Create a figure and store its handle
fig1 = figure()
PyPlot.plot(rand(10), label="First Plot in Figure 1")
title("Figure 1")
legend()

# Create another figure and store its handle
fig2 = figure()
PyPlot.plot(rand(10), label="First Plot in Figure 2")
title("Figure 2")
legend()

# Now, add a new plot to the specific figure (e.g., Figure 1)
figure(fig1.number)  # Activate Figure 1 by its handle
PyPlot.plot(rand(10), label="Second Plot in Figure 1")  # Plot on Figure 1
legend()  # Update legend to show all lines in Figure 1

# Similarly, to add to Figure 2, you can activate it by its handle
figure(fig2.number)  # Activate Figure 2
PyPlot.plot(rand(10), label="Second Plot in Figure 2")  # Plot on Figure 2
legend()  # Update legend to show all lines in Figure 2

