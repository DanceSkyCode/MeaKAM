import torch.nn as nn
class OutNet(nn.Module):
    def __init__(self, input_dim):
        super(OutNet, self).__init__()
        self.linear_1 = nn.Linear(input_dim, globals()['linear1_unit'], bias=True)
        self.linear_2 = nn.Linear(globals()['linear1_unit'], globals()['linear2_unit'] , bias=True)
        self.linear_3 = nn.Linear(globals()['linear2_unit'], globals()['linear3_unit'], bias=True)
        self.linear_4 = nn.Linear(globals()['linear3_unit'], 1, bias=True)
        self.dropout_layer = nn.Dropout(0.1)    
        for layer in [self.linear_1, self.linear_2]:
            nn.init.xavier_normal_(layer.weight)

    def forward(self, sequence, ele, height, weight, lens):
        # if len(self.high_level_locs) > 0:
        #     sequence = torch.cat((sequen
        # ce, others[:, :, self.high_level_locs]), dim=2)
        sequence = self.linear_1(sequence)
        sequence = self.dropout_layer(sequence)
        sequence = self.linear_2(sequence)
        sequence = self.dropout_layer(sequence)
        sequence = self.linear_3(sequence)
        sequence = self.dropout_layer(sequence)
        sequence = self.linear_4(sequence)        
        X = sequence.shape[0]
        Y = sequence.shape[1]
        sequence = sequence.view(X,Y)
        sequence = sequence / height / weight
        # weight = others[:, 0, WEIGHT].unsqueeze(1).unsqueeze(2)
        # height = others[:, 0, HEIGHT].unsqueeze(1).unsqueeze(2)
        # sequence = torch.div(sequence, weight * GRAVITY * height / 100)
        return sequence