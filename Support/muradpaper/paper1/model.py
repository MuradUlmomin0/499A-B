import torch
import torch.nn as nn

class SecureDyn1DCNN(nn.Module):
    def __init__(self, num_classes=11):
        super(SecureDyn1DCNN, self).__init__()
        
        # Shared Feature Extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # The Decoupled Classifiers
        self.global_classifier = nn.Linear(32, num_classes)
        self.personalized_classifier = nn.Linear(32, num_classes)

    def forward(self, x):
        features = self.feature_extractor(x)
        global_logits = self.global_classifier(features)
        pers_logits = self.personalized_classifier(features)
        return global_logits, pers_logits, features