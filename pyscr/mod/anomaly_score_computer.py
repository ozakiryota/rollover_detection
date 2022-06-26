import torch

def computeAnomalyScore(real_images, reconstracted_images, real_features, reconstracted_features, dis_loss_weight=0.1):
    res_loss = torch.abs(real_images - reconstracted_images)
    res_loss = res_loss.view(res_loss.size(0), -1)
    res_loss = torch.sum(res_loss, dim=1)

    dis_loss = torch.abs(real_features - reconstracted_features)
    dis_loss = dis_loss.view(dis_loss.size(0), -1)
    dis_loss = torch.sum(dis_loss, dim=1)

    score = (1 - dis_loss_weight) * res_loss + dis_loss_weight * dis_loss
    score = torch.mean(score)

    return score


def test():
    import sys
    sys.path.append('../')
    from mod.discriminator import Discriminator

    ## data
    batch_size = 5
    img_size = 112
    z_dim = 20
    images1 = torch.randn(batch_size, 3, img_size, img_size)
    images2 = torch.randn(batch_size, 3, img_size, img_size)
    z = torch.randn(batch_size, z_dim)
    ## feature
    dis_net = Discriminator(z_dim, img_size)
    _, features1 = dis_net(images1, z)
    _, features2 = dis_net(images2, z)
    ## score
    anomaly_score = computeAnomalyScore(images1, images2, features1, features2)
    ## debug
    print("anomaly_score.size() =", anomaly_score.size())
    print("anomaly_score =", anomaly_score)

if __name__ == '__main__':
    test()