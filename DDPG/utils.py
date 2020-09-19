def hard_update(target_model, source_model):
    """
        copy source model weight to target model weight  
        target = source  
        Args:  
            target_model: pytorch model  
            source_model: pytorch model  
    """
    for target_param, source_param in zip(target_model.parameters(), source_model.parameters()):
        target_param.data.copy_(source_param.data)


def soft_update(target_model, source_model, tau=0.001):
    """
        copy source model weight to target model weight  
        target = (1 - tau) * target + tau * source  
        Args:  
            target_model: pytorch model  
            source_model: pytorch model  
            tau: float
    """
    for target_param, source_param in zip(target_model.parameters(), source_model.parameters()):
        target_param.data.copy_(
            (1.0 - tau) * target_param.data + tau * source_param.data
        )
