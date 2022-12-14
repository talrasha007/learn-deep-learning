import torch
from huggingface_hub import hf_hub_download
from transformers import TimeSeriesTransformerForPrediction

file = hf_hub_download(
    repo_id="kashif/tourism-monthly-batch", filename="train-batch.pt", repo_type="dataset"
)
batch = torch.load(file)

model = TimeSeriesTransformerForPrediction.from_pretrained(
    "huggingface/time-series-transformer-tourism-monthly"
)

PATH = './tst.pth'
model.load_state_dict(torch.load(PATH))

with torch.no_grad():
    outputs = model.generate(
        past_values=batch["past_values"],
        past_time_features=batch["past_time_features"],
        past_observed_mask=batch["past_observed_mask"],
        static_categorical_features=batch["static_categorical_features"],
        static_real_features=batch["static_real_features"],
        future_time_features=batch["future_time_features"],
    )

mean_prediction = outputs.sequences.mean(dim=1)
print(outputs.sequences.shape, mean_prediction.shape, batch["future_values"].shape)
print("mean_prediction: ", mean_prediction[0])
print("ground truth: ", batch["future_values"][0])
