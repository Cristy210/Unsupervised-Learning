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

# ╔═╡ 69053198-c124-4d5d-8fa9-2d826e75dbfb
data_mat = reshape(data_tens, :, size(data_tens, 3))

# ╔═╡ c299af00-de1d-4a3b-8925-7e4a358defdc
matrices = [data_mat[:, (i-1)*10000+1:i*10000] for i in 1:6]

# ╔═╡ d9abcce0-a32f-4821-8011-b4f437c750b1
M_1 = matrices[1]

# ╔═╡ 75a960b8-f35a-4af1-bc9f-b1481e757930
col_norms = [norm(M_1[:, i]) for i in 1:size(M_1, 2)]

# ╔═╡ 8962452a-ef78-4e5f-8ee5-ab885c1bb363
M1_norm_vec = [M_1[:, i] ./ col_norms[i] for i in 1:size(M_1, 2)]

# ╔═╡ 5564ba9a-209b-4591-a425-10e1926d5c7f
M1_normalized = hcat(M1_norm_vec...)

# ╔═╡ 6d64957d-931c-4fc6-9726-c1d33282fadb
 Q = transpose(M1_normalized) * M1_normalized

# ╔═╡ 0139fa9a-0b8d-49ff-96c7-174f305fe512
with_theme() do
	fig = Figure(; size=(1000, 600))
	# tick_pos = cumsum([size(data_mat[i], 2) for i in 1:swatch_count])
	hx = Axis(fig[1, 1], aspect=DataAspect(), yreversed=true)
	hm = heatmap!(hx, Q, colormap=:viridis)
	Colorbar(fig[1, 2], hm)
	fig
end

# ╔═╡ 9857c797-0bdf-4311-8bc3-383682e725ab
A = [transpose(M_1[:, i]) * M_1[:, j] / (norm(M_1[:, i]) * norm(M_1[:, j])) for i in 1:size(M_1, 2), j in 1:size(M_1, 2)]

# ╔═╡ Cell order:
# ╠═c4cdcf24-7536-11ef-3e1d-c7bd325528e3
# ╠═ad252bff-49ec-474c-87cc-4bc87459efbd
# ╠═bc40de1d-87d9-48f9-b27d-c24fe48dbf54
# ╠═017406c9-240a-4c71-9077-dfe46ff3d596
# ╠═2f719a40-771e-4cbb-b6e9-7f3642d1eb93
# ╠═85330616-e425-432f-91f6-1587da2a6415
# ╠═07fe35ed-50d5-444d-8ac0-92b8a83bf6dc
# ╠═69053198-c124-4d5d-8fa9-2d826e75dbfb
# ╠═c299af00-de1d-4a3b-8925-7e4a358defdc
# ╠═d9abcce0-a32f-4821-8011-b4f437c750b1
# ╠═75a960b8-f35a-4af1-bc9f-b1481e757930
# ╠═8962452a-ef78-4e5f-8ee5-ab885c1bb363
# ╠═5564ba9a-209b-4591-a425-10e1926d5c7f
# ╠═6d64957d-931c-4fc6-9726-c1d33282fadb
# ╠═0139fa9a-0b8d-49ff-96c7-174f305fe512
# ╠═9857c797-0bdf-4311-8bc3-383682e725ab
