### A Pluto.jl notebook ###
# v0.19.46

using Markdown
using InteractiveUtils

# ╔═╡ c4cdcf24-7536-11ef-3e1d-c7bd325528e3
import Pkg; Pkg.activate(@__DIR__)

# ╔═╡ ad252bff-49ec-474c-87cc-4bc87459efbd
using AppleAccelerate, CairoMakie, LinearAlgebra, PlutoUI, Colors, Clustering, GZip

# ╔═╡ bc40de1d-87d9-48f9-b27d-c24fe48dbf54
FILENAME = joinpath(@__DIR__, "train-images.idx3-ubyte")

# ╔═╡ 017406c9-240a-4c71-9077-dfe46ff3d596
mnist = GZip.read(FILENAME)

# ╔═╡ 2f719a40-771e-4cbb-b6e9-7f3642d1eb93
image_data = mnist[17:end]

# ╔═╡ 85330616-e425-432f-91f6-1587da2a6415
data_tens = reshape(image_data, 28, 28, 60000)

# ╔═╡ 07fe35ed-50d5-444d-8ac0-92b8a83bf6dc
with_theme() do
	fig = Figure(; size=(600, 600))
	ax = Axis(fig[1, 1], aspect=DataAspect(), yreversed=true)
	img = image!(ax, data_tens[:, :, 201])
	fig
end

# ╔═╡ Cell order:
# ╠═c4cdcf24-7536-11ef-3e1d-c7bd325528e3
# ╠═ad252bff-49ec-474c-87cc-4bc87459efbd
# ╠═bc40de1d-87d9-48f9-b27d-c24fe48dbf54
# ╠═017406c9-240a-4c71-9077-dfe46ff3d596
# ╠═2f719a40-771e-4cbb-b6e9-7f3642d1eb93
# ╠═85330616-e425-432f-91f6-1587da2a6415
# ╠═07fe35ed-50d5-444d-8ac0-92b8a83bf6dc
