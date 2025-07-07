from services import CurrentRecommendationService
from metrics import precision_at_k, recall_at_k, f1_at_k
import random

service = CurrentRecommendationService()
all_users = service.get_available_users()
sample_users = random.sample(all_users, 100)  # 100 kullanıcı seç

precisions, recalls, f1s = [], [], []
for user_id in sample_users:
    user_history = service.get_user_game_history(user_id)
    true_games = [item['game']['id'] for item in user_history if item['recommended']]
    recommended = service.get_user_recommendations(user_id, n_recommendations=10)
    predicted_games = [item['id'] for item in recommended]
    p = precision_at_k(predicted_games, true_games, k=10)
    r = recall_at_k(predicted_games, true_games, k=10)
    f1 = f1_at_k(predicted_games, true_games, k=10)
    precisions.append(p)
    recalls.append(r)
    f1s.append(f1)

print(f"Ortalama Precision@10: {sum(precisions)/len(precisions):.2f}")
print(f"Ortalama Recall@10: {sum(recalls)/len(recalls):.2f}")
print(f"Ortalama F1@10: {sum(f1s)/len(f1s):.2f}")

# sample_users listesindeki ilk kullanıcı için detaylı inceleme
user_id = sample_users[0]
user_history = service.get_user_game_history(user_id)
true_games = [item['game']['id'] for item in user_history if item['recommended']]
recommended = service.get_user_recommendations(user_id, n_recommendations=10)
predicted_games = [item['id'] for item in recommended]

print(f"User ID: {user_id}")
print("True games (beğenilenler):", true_games)
print("Predicted games (önerilenler):", predicted_games)
print("Kesişim:", set(true_games) & set(predicted_games)) 