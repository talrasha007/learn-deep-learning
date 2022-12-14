import torch
import torch.optim as optim

from huggingface_hub import hf_hub_download
from transformers import TimeSeriesTransformerForPrediction

file = hf_hub_download(
    repo_id="kashif/tourism-monthly-batch", filename="train-batch.pt", repo_type="dataset"
)
batch = torch.load(file)

model = TimeSeriesTransformerForPrediction.from_pretrained(
    "huggingface/time-series-transformer-tourism-monthly"
)

optimizer = optim.Adam(model.parameters(), lr=0.0001)

for epoch in range(32):  # loop over the dataset multiple times

    # during training, one provides both past and future values
    # as well as possible additional features
    outputs = model(
        past_values=batch["past_values"],
        past_time_features=batch["past_time_features"],
        past_observed_mask=batch["past_observed_mask"],
        static_categorical_features=batch["static_categorical_features"],
        static_real_features=batch["static_real_features"],
        future_values=batch["future_values"],
        future_time_features=batch["future_time_features"],
    )

    loss = outputs.loss
    loss.backward()
    optimizer.step()

    # print statistics
    print(f'[{epoch + 1:2d}] loss: {loss.item() / 100:.3f}')

print('Finished Training')
PATH = './tst.pth'
torch.save(model.state_dict(), PATH)
