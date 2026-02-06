def load_best_model(teacher_model):
    
    best_model_path = args.teacher_bestmodel_path
    checkpoint = torch.load(best_model_path)
    
    teacher_model.load_state_dict(checkpoint['state_dict'])
    
    teacher_model.eval()
    
    return teacher_model