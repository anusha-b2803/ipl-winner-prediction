import sqlite3
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from models.transformer_model import build_model

DB_PATH = os.getenv("SQLITE_DB_PATH", "./data/ipl_stats.db")
MODEL_PATH = "./models/weights.pt"

def load_training_data():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    # Get all seasons up to 2025
    c.execute("SELECT DISTINCT year FROM team_season_stats WHERE year <= 2025 ORDER BY year")
    rows_years = c.fetchall()
    years = [r[0] for r in rows_years]
    
    dataset = []
    
    # Track title counts dynamically to avoid data leakage
    title_counts = {}

    for year in years:
        c.execute("SELECT * FROM team_season_stats WHERE year = ? ORDER BY position", (year,))
        teams = [dict(r) for r in c.fetchall()]
        
        if not teams:
            continue
            
        # Features: wins, losses, nrr, points, position, historical_titles, qualified_playoffs
        # We need a fixed sequence length (max 10 teams)
        max_teams = 10
        year_features = []
        target_idx = -1
        
        # Merge duplicate RCB entries if they exist
        team_map = {}
        for t in teams:
            name = t['team'].replace('Royal Challengers Bangalore', 'Royal Challengers Bengaluru')
            if name not in team_map or t['points'] > team_map[name]['points']:
                team_map[name] = t
        
        unique_teams = list(team_map.values())
        # Truncate to 10 if somehow more exist (data artifacts)
        unique_teams = unique_teams[:max_teams]
        
        for i, t in enumerate(unique_teams):
            team_name = t['team'].replace('Royal Challengers Bangalore', 'Royal Challengers Bengaluru')
            h_titles = title_counts.get(team_name, 0)
            
            # Normalize basic features
            feat = [
                t['wins'] / 16.0,           # Matches vary, 16 is a safe max
                t['losses'] / 16.0,
                t['nrr'],                    # NRR is already small (-2 to +2 usually)
                t['points'] / 32.0,          # Max points usually ~25-28
                t['position'] / 10.0,
                h_titles / 5.0,              # Max titles is 5
                1.0 if t['qualified_playoffs'] else 0.0
            ]
            year_features.append(feat)
            
            if t['won_title']:
                target_idx = i
                
        # Update title counts for next year based on current winner
        # Note: We do this AFTER processing the current year to ensure leakage-free historical count
        c.execute("SELECT team FROM team_season_stats WHERE year = ? AND won_title = 1", (year,))
        winner_row = c.fetchone()
        if winner_row:
            w_name = winner_row[0]
            title_counts[w_name] = title_counts.get(w_name, 0) + 1

        # Padding if fewer than 10 teams (early seasons had 8 or 9)
        while len(year_features) < max_teams:
            year_features.append([0.0] * 7)
            
        if target_idx != -1:
            dataset.append({
                "features": year_features,
                "label": target_idx
            })
            
    conn.close()
    return dataset

def train():
    print("Preparing training data from SQLite...")
    data = load_training_data()
    if not data:
        print("No training data found! Synchronize history first.")
        return

    # Convert to tensors
    X = torch.tensor([d['features'] for d in data], dtype=torch.float32)
    y = torch.tensor([d['label'] for d in data], dtype=torch.long)
    
    print(f"Dataset loaded: {len(X)} seasons of history.")

    # Initialize model
    model = build_model(input_dim=7)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    print("Starting training...")
    model.train()
    epochs = 500
    
    for epoch in range(epochs):
        # We have a small dataset, so we can train on full batch
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 50 == 0:
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            accuracy = (predicted == y).sum().item() / y.size(0)
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.2%}")

    print(f"Training complete. Saving weights to {MODEL_PATH}")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)

if __name__ == "__main__":
    train()
