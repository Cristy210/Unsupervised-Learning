### A Pluto.jl notebook ###
# v0.19.46

using Markdown
using InteractiveUtils

# ╔═╡ 3cb4a193-3f36-42b5-9227-109805a2c624
import Pkg; Pkg.activate(@__DIR__)

# ╔═╡ cf72f8ad-db76-4142-9a87-d93f6fa6ba17
using CSV, DataFrames, LinearAlgebra, Clustering, ArnoldiMethod, ProgressLogging, Logging, Random

# ╔═╡ 7de038d6-c135-11ef-039b-75d27586a5d9


# ╔═╡ 48bd5115-a4c3-43c5-9b8d-92a9133f388f
html"""<style>
input[type*="range"] {
	width: calc(100% - 4rem);
}
main {
    max-width: 96%;
    margin-left: 0%;
    margin-right: 2% !important;
}
"""

# ╔═╡ 372370d7-c3df-4251-9415-73e64e3f5796
Data_dir = joinpath(@__DIR__, "Datasets", "data.csv")

# ╔═╡ c98bb3ae-e9c4-49d4-a8ce-0b822d26f9b6
Data_mat = CSV.read(Data_dir, DataFrame)

# ╔═╡ f303fe2d-f836-490c-84e6-ae43ce74a9f5
patient_ids = Data_mat[:, :id]

# ╔═╡ 465d0303-e8fb-41ed-a8f3-b36c1c72d6c4
feature_names = names(Data_mat)[3:32]

# ╔═╡ 425dff93-543c-4523-9a3b-06137324e7b9
features_matrix = Matrix(Data_mat[:, feature_names])'

# ╔═╡ e09c33b2-7a56-425a-b0b4-4d58d8bfc980
col_norms = [norm(features_matrix[:, i]) for i in 1:size(features_matrix, 2)]

# ╔═╡ c2bef7aa-6366-40ae-88c3-7e9c4d84848b
Norm_vec = [features_matrix[:, i] ./ col_norms[i] for i in 1:size(features_matrix, 2)]

# ╔═╡ 06620659-4dbf-4e9b-a1df-a7ba501bbe68
Norm_mat = hcat(Norm_vec...)

# ╔═╡ b65290e1-a2fc-4de8-bf27-0e94de33d9f2
A = transpose(Norm_mat) * Norm_mat

# ╔═╡ fb8f6e22-1aae-4bd8-8ffb-edf721a7c5bb
md"""
### Similarity Matrix
"""

# ╔═╡ 4d6c8a1e-85a1-4dee-a6d9-9a5cee141826
S = exp.((-2 .* acos.(clamp.(A, -1, 1))))


# ╔═╡ 5efbfe7f-8119-4882-972c-427d64206bb2
diag_mat = Diagonal(vec(sum(S, dims=2)))

# ╔═╡ e1437b22-a1f2-4e85-97d1-61dd5241596d
D_sqrinv = sqrt(inv(diag_mat))

# ╔═╡ e0b54050-3098-431a-9644-9358212e805e
md"""
### Laplacian Matrix
"""

# ╔═╡ f65e43ff-6ff7-4931-9a6f-4c87c4424599
L_sym = Symmetric(I - (D_sqrinv * S * D_sqrinv))

# ╔═╡ 5323766b-eae5-4513-a9e0-4f73a6e35fe5
decomp, history = partialschur(L_sym; nev=(2), which=:SR)

# ╔═╡ 2a1b3375-4b54-4adb-b1ee-d1e26698d310
_, EV = partialeigen(decomp)

# ╔═╡ 8733553e-ffe4-47fc-a138-830b03f38e7c
row_norms = [norm(EV[i, :]) for i in 1:size(EV, 1)]

# ╔═╡ a5c2586e-af4b-4ede-821b-89cbd0a828cd
T = EV ./ row_norms

# ╔═╡ e787fff0-0a1f-473c-a1e7-3daf1f48bf6f
function batchkmeans(X, k, args...; nruns=200, kwargs...)
	runs = @withprogress map(1:nruns) do idx
		# Run K-means
		Random.seed!(idx)  # set seed for reproducibility
		result = with_logger(NullLogger()) do
			kmeans(X, k, args...; kwargs...)
		end

		# Log progress and return result
		@logprogress idx/nruns
		return result
	end

	# Print how many converged
	nconverged = count(run -> run.converged, runs)
	@info "$nconverged/$nruns runs converged"

	# Return runs sorted best to worst
	return sort(runs; by=run->run.totalcost)
end

# ╔═╡ 4696fb91-95ad-4bb9-8673-49e043b1e76a
spec_clusterings = batchkmeans(permutedims(T), 2; maxiter=1000)

# ╔═╡ 99454533-f332-4f5c-ad1d-0d60211a06c9
clu_assignments = spec_clusterings[1].assignments

# ╔═╡ 76e520ad-215c-495e-bf4a-1ca883dccc9b
labels = Data_mat[:, :diagnosis]

# ╔═╡ 9dac3b4c-0067-4a5d-aa62-1adfe2006ca4
l1 = findall(.==(1), clu_assignments)

# ╔═╡ c415fcd2-22e9-442f-97ee-9f24b42d5173
l2 = findall(.==(2), clu_assignments)

# ╔═╡ 8d3603a5-ba2c-4e90-b64d-3594eec56981
l1_count = count(.==(1), clu_assignments)

# ╔═╡ 1b9e2fb0-c091-4d9d-b726-6fba892dd0a1
l2_count = count(.==(2), clu_assignments)

# ╔═╡ ffcc7965-16eb-43d0-ae1f-d8dc065a51d2
l1_labels = labels[l1]

# ╔═╡ 7f574073-2848-4d70-8e48-63a848fe5a6a
l2_labels = labels[l2]

# ╔═╡ 7302ea9e-362b-4e96-a743-ba5ee7b4cf45
unique(l1_labels)

# ╔═╡ c3a3b12f-a489-4cba-a642-01be1eae4d39
function count_labels(labels)
    unique_labels = unique(labels)
    counts = Dict(label => count(x -> x == label, labels) for label in unique_labels)
    return counts
end

# ╔═╡ 68221ea3-40ca-40a2-898b-67ec842c4b89
l1_counts = count_labels(l1_labels)

# ╔═╡ 75bca777-b78a-49f0-ad97-054de0e53d9b
l2_counts = count_labels(l2_labels)

# ╔═╡ cec77293-4155-44be-9055-e018bffcf1c6
total_l1 = sum(values(l1_counts))

# ╔═╡ 899b0997-d150-4da9-bb61-d6a244dc2103
md"""
##### Calculate percentages for l1_labels
"""

# ╔═╡ 36dbe1fc-3365-4922-90f8-b3833251e887
l1_percentages = Dict(label => (count / total_l1) * 100 for (label, count) in l1_counts)

# ╔═╡ 504bd8f5-be2c-48d8-b900-1c3636fd28e0
md"""
##### Calculate percentages for l2_labels
"""

# ╔═╡ 343f262e-a080-4738-b67b-fb84b1779e92
total_l2 = sum(values(l2_counts))

# ╔═╡ 4685de83-b3c4-42c1-acd3-1a83aa19f225
l2_percentages = Dict(label => (count / total_l2) * 100 for (label, count) in l2_counts)

# ╔═╡ Cell order:
# ╠═7de038d6-c135-11ef-039b-75d27586a5d9
# ╠═48bd5115-a4c3-43c5-9b8d-92a9133f388f
# ╠═3cb4a193-3f36-42b5-9227-109805a2c624
# ╠═cf72f8ad-db76-4142-9a87-d93f6fa6ba17
# ╠═372370d7-c3df-4251-9415-73e64e3f5796
# ╠═c98bb3ae-e9c4-49d4-a8ce-0b822d26f9b6
# ╠═f303fe2d-f836-490c-84e6-ae43ce74a9f5
# ╠═465d0303-e8fb-41ed-a8f3-b36c1c72d6c4
# ╠═425dff93-543c-4523-9a3b-06137324e7b9
# ╠═e09c33b2-7a56-425a-b0b4-4d58d8bfc980
# ╠═c2bef7aa-6366-40ae-88c3-7e9c4d84848b
# ╠═06620659-4dbf-4e9b-a1df-a7ba501bbe68
# ╠═b65290e1-a2fc-4de8-bf27-0e94de33d9f2
# ╟─fb8f6e22-1aae-4bd8-8ffb-edf721a7c5bb
# ╠═4d6c8a1e-85a1-4dee-a6d9-9a5cee141826
# ╠═5efbfe7f-8119-4882-972c-427d64206bb2
# ╠═e1437b22-a1f2-4e85-97d1-61dd5241596d
# ╟─e0b54050-3098-431a-9644-9358212e805e
# ╠═f65e43ff-6ff7-4931-9a6f-4c87c4424599
# ╠═5323766b-eae5-4513-a9e0-4f73a6e35fe5
# ╠═2a1b3375-4b54-4adb-b1ee-d1e26698d310
# ╠═8733553e-ffe4-47fc-a138-830b03f38e7c
# ╠═a5c2586e-af4b-4ede-821b-89cbd0a828cd
# ╠═e787fff0-0a1f-473c-a1e7-3daf1f48bf6f
# ╠═4696fb91-95ad-4bb9-8673-49e043b1e76a
# ╠═99454533-f332-4f5c-ad1d-0d60211a06c9
# ╠═76e520ad-215c-495e-bf4a-1ca883dccc9b
# ╠═9dac3b4c-0067-4a5d-aa62-1adfe2006ca4
# ╠═c415fcd2-22e9-442f-97ee-9f24b42d5173
# ╠═8d3603a5-ba2c-4e90-b64d-3594eec56981
# ╠═1b9e2fb0-c091-4d9d-b726-6fba892dd0a1
# ╠═ffcc7965-16eb-43d0-ae1f-d8dc065a51d2
# ╠═7f574073-2848-4d70-8e48-63a848fe5a6a
# ╠═7302ea9e-362b-4e96-a743-ba5ee7b4cf45
# ╠═c3a3b12f-a489-4cba-a642-01be1eae4d39
# ╠═68221ea3-40ca-40a2-898b-67ec842c4b89
# ╠═75bca777-b78a-49f0-ad97-054de0e53d9b
# ╠═cec77293-4155-44be-9055-e018bffcf1c6
# ╟─899b0997-d150-4da9-bb61-d6a244dc2103
# ╠═36dbe1fc-3365-4922-90f8-b3833251e887
# ╟─504bd8f5-be2c-48d8-b900-1c3636fd28e0
# ╠═343f262e-a080-4738-b67b-fb84b1779e92
# ╠═4685de83-b3c4-42c1-acd3-1a83aa19f225
