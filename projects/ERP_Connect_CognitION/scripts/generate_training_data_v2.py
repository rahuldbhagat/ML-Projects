"""
Training Data Generator v2.0 for ERP Connect CogitION
Generates 1500 high-quality Q&A samples from 30 seed pairs
Uses template-based variation for diversity
"""

import json
import random
from pathlib import Path
from datetime import datetime
import yaml

class TrainingDataGenerator:
    """Generate training data with smart variations"""
    
    def __init__(self, config_path='configs/data_generation_config.yaml'):
        # Load config or use defaults
        self.config = self.get_default_config()
        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                self.config.update(yaml.safe_load(f))
        
        self.seed_file = Path(self.config['input']['seed_file'])
        self.output_dir = Path(self.config['output']['processed_dir'])
        self.target_samples = self.config['generation']['target_samples']
        self.variations_per_seed = self.config['generation']['variations_per_seed']
        
        # Variation templates
        self.question_templates = self.get_question_templates()
        self.code_variations = self.get_code_variations()
        
        self.stats = {
            'total_generated': 0,
            'by_category': {},
            'by_difficulty': {},
            'generation_time': None
        }
    
    def get_default_config(self):
        """Default configuration"""
        return {
            'input': {'seed_file': 'data/seed_qa_pairs.json'},
            'output': {'processed_dir': 'data/processed'},
            'generation': {
                'target_samples': 1500,
                'variations_per_seed': 50
            },
            'split': {
                'train_ratio': 0.8,
                'val_ratio': 0.1,
                'test_ratio': 0.1,
                'random_seed': 42
            }
        }
    
    def get_question_templates(self):
        """Question variation templates"""
        return {
            'direct': [
                "{question}",
                "{question}?",
                "How to {action}",
                "What is the solution for {problem}",
                "Steps to {action}"
            ],
            'troubleshooting': [
                "Error: {problem}",
                "Getting error: {problem}",
                "Troubleshooting: {problem}",
                "{problem} - how to fix?",
                "Help with {problem}"
            ],
            'configuration': [
                "How to configure {component}",
                "Setting up {component}",
                "Configuration for {component}",
                "{component} setup guide",
                "Best practices for {component}"
            ],
            'help_seeking': [
                "Need help with {problem}",
                "Can you help me {action}",
                "I'm having trouble with {problem}",
                "Assistance needed: {problem}",
                "Please help: {problem}"
            ],
            'procedural': [
                "What are the steps to {action}",
                "How do I {action}",
                "Guide for {action}",
                "Procedure to {action}",
                "Instructions for {action}"
            ],
            'error_based': [
                "Server responded with {error_code}",
                "HTTP {error_code} error",
                "Getting {error_code} status",
                "{error_code} error troubleshooting",
                "Fix for {error_code}"
            ],
            'system_based': [
                "{component} won't start",
                "{component} not working",
                "{component} issues",
                "{component} troubleshooting",
                "Problem with {component}"
            ],
            'urgent': [
                "Urgent: {problem}",
                "Production issue: {problem}",
                "Critical: {problem}",
                "{problem} (urgent)",
                "Emergency: {problem}"
            ]
        }
    
    def get_code_variations(self):
        """Code/identifier variations"""
        return {
            'cost_centers': ['ENERGY572', 'ENERGY001', 'ENERGY999', 'TRADE456', 'AMEX123', 'POWER888', 'GAS777'],
            'gl_accounts': ['400000', '500000', '600000', '700000', '410000', '510000'],
            'company_codes': ['1000', '2000', '3000', '1100', '2100'],
            'business_units': ['AMEX001', 'AMEX002', 'TRADE001', 'ENERGY001', 'POWER001'],
            'trade_ids': ['TRD-12345', 'TRD-67890', 'TRD-11111', 'TRD-99999', 'TRD-55555'],
            'profit_centers': ['PC001', 'PC002', 'PC_ENERGY', 'PC_POWER', 'PC_GAS'],
            'http_codes': ['400', '401', '403', '404', '500', '502', '503', '504']
        }
    
    def generate_all(self):
        """Generate all training samples"""
        print("="*60)
        print("TRAINING DATA GENERATION v2.0")
        print("="*60)
        
        start_time = datetime.now()
        
        # Load seeds
        print(f"\n📖 Loading seed Q&A pairs from: {self.seed_file}")
        with open(self.seed_file, 'r') as f:
            seeds = json.load(f)
        print(f"   Loaded {len(seeds)} seed pairs")
        
        # Generate variations
        print(f"\n🔄 Generating {self.variations_per_seed} variations per seed...")
        print(f"   Target: {self.target_samples} total samples")
        
        all_samples = []
        for seed_id, seed in enumerate(seeds):
            variations = self.generate_variations(seed, seed_id)
            all_samples.extend(variations)
            
            if (seed_id + 1) % 10 == 0:
                print(f"   Progress: {seed_id + 1}/{len(seeds)} seeds processed")
        
        print(f"\n✅ Generated {len(all_samples)} samples")
        
        # Split data
        print(f"\n📊 Splitting data...")
        train, val, test = self.split_data(all_samples)
        
        # Save splits
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_split(train, 'train.jsonl')
        self.save_split(val, 'validation.jsonl')
        self.save_split(test, 'test.jsonl')
        
        # Save statistics
        end_time = datetime.now()
        self.stats['total_generated'] = len(all_samples)
        self.stats['generation_time'] = str(end_time - start_time)
        self.save_stats()
        
        print("\n" + "="*60)
        print("✅ TRAINING DATA GENERATION COMPLETE!")
        print("="*60)
        print(f"Total samples: {len(all_samples)}")
        print(f"Train: {len(train)} samples")
        print(f"Validation: {len(val)} samples")
        print(f"Test: {len(test)} samples")
        print(f"Time: {self.stats['generation_time']}")
        print(f"\nSaved to: {self.output_dir}")
    
    def generate_variations(self, seed, seed_id):
        """Generate variations for a single seed"""
        variations = []
        prompt_base = seed['prompt']
        response_base = seed['response']
        category = seed.get('category', 'GENERAL')
        component = seed.get('component', 'SYSTEM')
        difficulty = seed.get('difficulty', 'intermediate')
        
        # Update stats
        if category not in self.stats['by_category']:
            self.stats['by_category'][category] = 0
        if difficulty not in self.stats['by_difficulty']:
            self.stats['by_difficulty'][difficulty] = 0
        
        # Generate variations
        for var_id in range(self.variations_per_seed):
            # Apply variation strategy
            if var_id == 0:
                # First variation is original
                prompt = prompt_base
                response = response_base
            else:
                # Generate variations
                prompt = self.vary_prompt(prompt_base, seed, var_id)
                response = self.vary_response(response_base, seed, var_id)
            
            sample = {
                'prompt': prompt,
                'response': response,
                'category': category,
                'component': component,
                'difficulty': difficulty,
                'seed_id': seed_id,
                'variation_id': var_id
            }
            
            variations.append(sample)
            self.stats['by_category'][category] += 1
            self.stats['by_difficulty'][difficulty] += 1
        
        return variations
    
    def vary_prompt(self, prompt, seed, var_id):
        """Generate prompt variation"""
        # Extract key elements
        category = seed.get('category', 'GENERAL')
        
        # Choose variation strategy based on var_id
        strategy_idx = var_id % 8
        
        if strategy_idx == 0:
            # Direct question
            return prompt
        elif strategy_idx == 1:
            # Remove question mark
            return prompt.rstrip('?')
        elif strategy_idx == 2:
            # Add urgency
            return f"Urgent: {prompt}"
        elif strategy_idx == 3:
            # Add context
            return f"Production issue: {prompt}"
        elif strategy_idx == 4:
            # Rephrase as help request
            return f"Need help: {prompt}"
        elif strategy_idx == 5:
            # Add system context
            return f"In CTRM-SAP integration: {prompt}"
        elif strategy_idx == 6:
            # Make more specific
            return prompt.replace("How to", "What are the steps to")
        else:
            # Replace codes if present
            varied = prompt
            for code_type, codes in self.code_variations.items():
                for code in codes:
                    if code in prompt:
                        # Replace with different code from same type
                        new_code = random.choice([c for c in codes if c != code])
                        varied = varied.replace(code, new_code, 1)
                        break
            return varied if varied != prompt else prompt
    
    def vary_response(self, response, seed, var_id):
        """Generate response variation"""
        # For most variations, keep response same (model learns patterns)
        # Only vary codes/identifiers to match prompt
        
        varied = response
        
        # Replace codes to match varied prompt
        if var_id % 8 == 7:  # Code variation
            for code_type, codes in self.code_variations.items():
                for code in codes:
                    if code in response:
                        # Use consistent replacement
                        idx = (var_id // 8) % len(codes)
                        new_code = codes[idx]
                        varied = varied.replace(code, new_code)
                        break
        
        return varied
    
    def split_data(self, samples):
        """Split data into train/val/test sets"""
        random.seed(self.config['split']['random_seed'])
        
        # Shuffle
        shuffled = samples.copy()
        random.shuffle(shuffled)
        
        # Calculate split sizes
        total = len(shuffled)
        train_size = int(total * self.config['split']['train_ratio'])
        val_size = int(total * self.config['split']['val_ratio'])
        
        # Split
        train = shuffled[:train_size]
        val = shuffled[train_size:train_size + val_size]
        test = shuffled[train_size + val_size:]
        
        print(f"   Train: {len(train)} samples ({len(train)/total*100:.1f}%)")
        print(f"   Val: {len(val)} samples ({len(val)/total*100:.1f}%)")
        print(f"   Test: {len(test)} samples ({len(test)/total*100:.1f}%)")
        
        return train, val, test
    
    def save_split(self, samples, filename):
        """Save data split as JSONL"""
        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            for sample in samples:
                f.write(json.dumps(sample) + '\n')
        print(f"   ✓ Saved: {filename}")
    
    def save_stats(self):
        """Save generation statistics"""
        stats_file = self.output_dir / 'generation_report.json'
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
        print(f"\n📊 Statistics saved to: {stats_file}")

def main():
    """Main entry point"""
    generator = TrainingDataGenerator()
    generator.generate_all()

if __name__ == '__main__':
    main()
