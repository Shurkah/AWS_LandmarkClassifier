import torch
import torch.nn as nn

# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 50, dropout: float = 0.7):

        super(MyModel, self).__init__()

        self.model = nn.Sequential(
            
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Debugging: got rid of all dropouts in convo layers
            # Test with removed stride = 3; way better, leave like this
                        
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Three convoluted layers seem to be insufficient, 
            # the model is not learning
            # Adding two more
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, padding=1),  
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  

            nn.Flatten(),

            # Maybe my input_to_head is getting me into trouble, 
            # let us try a straightforward architecture without it

            nn.Linear(256*7*7, 1024),  
            nn.Dropout(dropout),
            nn.BatchNorm1d(1024),
            nn.ReLU(),

            nn.Linear(1024, 512),  
            nn.Dropout(dropout),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, num_classes)
        )

    # And then I aldo need to update my forward pass
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        return self.model(x)


        # I am tired of calculating and recalculating tensor sizes, 
        # so I just make the model calculate output tensor dimensions here.
        # Anyway, the model has to operate with variables.

        # self.input_to_head = self.get_input_to_head()

        # self.linear_layers = nn.Sequential(
        #    nn.Linear(self.input_to_head, num_classes * 2),
        #    nn.Dropout(dropout),
        #    nn.ReLU(),
        #    nn.Linear(num_classes * 2, num_classes),
        #    nn.Softmax(dim=1) 
        #)

    # I will still leave the function here in case I get back to the previous arch
    # no need to comment it out as it is not called
    def get_input_to_head(self):
        test_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            x = self.model(test_input)
        return x.view(x.size(0), -1).size(1)

    # Here is my forward pass for the previous architecture
    # def forward(self, x):
    #    x = self.model(x)
    #    x = x.view(x.size(0), -1)
    #    x = self.linear_layers(x)
    #    return x

# Create an instance of the model
model = MyModel(num_classes=50)
print(model)


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = next(dataiter)

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
