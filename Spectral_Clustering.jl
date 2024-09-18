### A Pluto.jl notebook ###
# v0.19.46

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

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

# ╔═╡ ff2b2b9e-9e2a-4770-aa38-0f4adfb9d7eb
md"""
### Adjacency Matrix
"""

# ╔═╡ 6d64957d-931c-4fc6-9726-c1d33282fadb
 Q = transpose(M1_normalized) * M1_normalized

# ╔═╡ 877aa397-8c47-4c38-8ede-8a9a4644147a
S = exp.((-2 .* acos.(clamp.(Q, -1, 1))))

# ╔═╡ 0139fa9a-0b8d-49ff-96c7-174f305fe512
with_theme() do
	fig = Figure(; size=(1000, 700))
	# tick_pos = cumsum([size(data_mat[i], 2) for i in 1:swatch_count])
	hx = Axis(fig[1, 1], aspect=DataAspect(), yreversed=true, title = "Cosine Similarity", titlesize = 18)
	hm = heatmap!(hx, S, colormap=:viridis)
	Colorbar(fig[1, 2], hm)
	fig
end

# ╔═╡ c28ae100-54a6-45ab-9d86-c1f4b0851b2a
diag_mat = diagm(vec(sum(S, dims=2)))

# ╔═╡ 192c8fee-2ea0-4b67-a861-e8c0302a02d7
D_sqrinv = sqrt(inv(diag_mat))

# ╔═╡ a2b31a16-b6da-4d52-af9a-8a8beaec54d0
L_sym = Symmetric(I - (D_sqrinv * S * D_sqrinv))

# ╔═╡ 90fc49b1-3079-4df5-abe0-625059edadb0
EV = eigen(L_sym).vectors

# ╔═╡ 2739eff5-f580-4a83-881b-ae6159781679
EV_lin = EV[:, 1:10]

# ╔═╡ 88eeb3de-a048-43eb-993a-3f1e731f9d21
T_lin = EV_lin ./ [norm(EV_lin[i, :]) for i in 1:size(EV_lin, 1)]

# ╔═╡ 7afe236a-56e1-48fd-b906-f778c145fc26
min_cost = Inf64; best_clustering = nothing

# ╔═╡ 9174df36-f167-4a1b-b46a-22f3615ede45
@bind KM_Runs PlutoUI.Slider(100:100:2500; show_value=true)

# ╔═╡ de554a82-0e85-4198-85ed-36540be877bd
kmeansall = [[kmeans(permutedims(T_lin), 10; maxiter=1000)] for _ in 1:KM_Runs]

# ╔═╡ 45dcd318-dcfc-4e5a-af88-c2149bad66cb
min_index = argmin(kmeansall[i][1].totalcost for i in 1:KM_Runs)

# ╔═╡ 9bef3679-42bb-4cd9-bef5-fd09b01bd301
M_1[1, :]

# ╔═╡ 2b78c138-4433-4d4e-9030-ab9a2b5c114d
clusters_lin = kmeansall[min_index][1].assignments

# ╔═╡ fe3cc822-95f4-4359-bfb8-bdc039efcc84
M1_10 = [reshape(M_1[:, i], 28, 28) for i in 1:10]

# ╔═╡ 1a9518df-0c0c-4519-9435-8ff90ed73ff3
M1_10_mat = reshape(M1_10, 5, 2)

# ╔═╡ 3b96f699-5150-477a-8fd5-cb9a52707b06
with_theme() do
	fig = Figure(; size=(800, 600))
	for ROW in 1:5, COL in 1:2
		ax = Axis(fig[ROW, COL], aspect = DataAspect(), yreversed = true)
		image!(ax, M1_10_mat[ROW, COL])
	end
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
# ╠═69053198-c124-4d5d-8fa9-2d826e75dbfb
# ╠═c299af00-de1d-4a3b-8925-7e4a358defdc
# ╠═d9abcce0-a32f-4821-8011-b4f437c750b1
# ╠═75a960b8-f35a-4af1-bc9f-b1481e757930
# ╠═8962452a-ef78-4e5f-8ee5-ab885c1bb363
# ╠═5564ba9a-209b-4591-a425-10e1926d5c7f
# ╟─ff2b2b9e-9e2a-4770-aa38-0f4adfb9d7eb
# ╠═6d64957d-931c-4fc6-9726-c1d33282fadb
# ╠═877aa397-8c47-4c38-8ede-8a9a4644147a
# ╠═0139fa9a-0b8d-49ff-96c7-174f305fe512
# ╠═c28ae100-54a6-45ab-9d86-c1f4b0851b2a
# ╠═192c8fee-2ea0-4b67-a861-e8c0302a02d7
# ╠═a2b31a16-b6da-4d52-af9a-8a8beaec54d0
# ╠═90fc49b1-3079-4df5-abe0-625059edadb0
# ╠═2739eff5-f580-4a83-881b-ae6159781679
# ╠═88eeb3de-a048-43eb-993a-3f1e731f9d21
# ╠═7afe236a-56e1-48fd-b906-f778c145fc26
# ╠═9174df36-f167-4a1b-b46a-22f3615ede45
# ╠═de554a82-0e85-4198-85ed-36540be877bd
# ╠═45dcd318-dcfc-4e5a-af88-c2149bad66cb
# ╠═9bef3679-42bb-4cd9-bef5-fd09b01bd301
# ╟─2b78c138-4433-4d4e-9030-ab9a2b5c114d
# ╠═fe3cc822-95f4-4359-bfb8-bdc039efcc84
# ╠═1a9518df-0c0c-4519-9435-8ff90ed73ff3
# ╠═3b96f699-5150-477a-8fd5-cb9a52707b06
