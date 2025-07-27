from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.scrollview import ScrollView
from kivy.uix.popup import Popup
from kivy.uix.progressbar import ProgressBar
from kivy.core.window import Window
from kivy.clock import Clock
from kivy.storage.jsonstore import JsonStore
import pandas as pd
import joblib
import numpy as np
import sqlite3
import datetime
import os
from threading import Thread

class DiabetesPredictionApp(App):
    def build(self):
        # Set window size for mobile-like experience
        Window.size = (400, 700)
        
        # Load the model
        self.model = joblib.load('src/diabetes_model.pkl')
        self.features = ['Gender', 'AGE', 'Urea', 'Cr', 'HbA1c', 'Chol', 'TG', 'HDL', 'LDL', 'VLDL', 'BMI']
        
        # Initialize storage
        self.store = JsonStore('diabetes_predictions.json')
        
        # Create main layout
        main_layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        
        # Title
        title = Label(
            text='Diabetes Prediction AI',
            size_hint_y=None,
            height=50,
            font_size='24sp',
            bold=True,
            color=(0.2, 0.6, 1, 1)
        )
        main_layout.add_widget(title)
        
        # Create scrollable form
        scroll_layout = GridLayout(cols=1, spacing=10, size_hint_y=None)
        scroll_layout.bind(minimum_height=scroll_layout.setter('height'))
        
        # Add form fields
        self.inputs = {}
        for feature in self.features:
            # Feature label
            label = Label(
                text=f'{feature}:',
                size_hint_y=None,
                height=30,
                halign='left',
                color=(0.3, 0.3, 0.3, 1)
            )
            scroll_layout.add_widget(label)
            
            # Input field
            text_input = TextInput(
                multiline=False,
                size_hint_y=None,
                height=40,
                hint_text=f'Enter {feature}',
                input_filter='float'
            )
            self.inputs[feature] = text_input
            scroll_layout.add_widget(text_input)
        
        # Scroll view
        scroll_view = ScrollView(size_hint=(1, 1))
        scroll_view.add_widget(scroll_layout)
        main_layout.add_widget(scroll_view)
        
        # Buttons
        button_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=50, spacing=10)
        
        predict_btn = Button(
            text='Predict',
            size_hint_x=0.5,
            background_color=(0.2, 0.8, 0.2, 1),
            on_press=self.predict
        )
        button_layout.add_widget(predict_btn)
        
        history_btn = Button(
            text='History',
            size_hint_x=0.5,
            background_color=(0.2, 0.6, 1, 1),
            on_press=self.show_history
        )
        button_layout.add_widget(history_btn)
        
        main_layout.add_widget(button_layout)
        
        # Progress bar (hidden initially)
        self.progress = ProgressBar(max=100, size_hint_y=None, height=20)
        self.progress.opacity = 0
        main_layout.add_widget(self.progress)
        
        # Results label
        self.results_label = Label(
            text='',
            size_hint_y=None,
            height=100,
            text_size=(Window.width - 20, None),
            halign='left',
            valign='top'
        )
        main_layout.add_widget(self.results_label)
        
        return main_layout
    
    def predict(self, instance):
        """Make prediction with validation"""
        try:
            # Show progress
            self.progress.opacity = 1
            self.progress.value = 0
            
            # Validate inputs
            values = []
            for feature in self.features:
                value = self.inputs[feature].text.strip()
                if not value:
                    self.show_error(f'Please enter {feature}')
                    return
                try:
                    values.append(float(value))
                except ValueError:
                    self.show_error(f'{feature} must be a number')
                    return
            
            # Update progress
            self.progress.value = 50
            
            # Make prediction
            values_df = pd.DataFrame([values], columns=self.features)
            prediction = self.model.predict(values_df)[0]
            
            # Update progress
            self.progress.value = 100
            
            # Show results
            result_text = f'Prediction: Class {prediction}\n\n'
            result_text += 'Input Values:\n'
            for feature, value in zip(self.features, values):
                result_text += f'{feature}: {value}\n'
            
            self.results_label.text = result_text
            self.results_label.color = (0.2, 0.8, 0.2, 1)
            
            # Save to history
            self.save_prediction(values, prediction)
            
            # Hide progress
            Clock.schedule_once(lambda dt: setattr(self.progress, 'opacity', 0), 1)
            
        except Exception as e:
            self.show_error(f'Prediction error: {str(e)}')
            self.progress.opacity = 0
    
    def save_prediction(self, values, prediction):
        """Save prediction to local storage"""
        timestamp = datetime.datetime.now().isoformat()
        prediction_data = {
            'timestamp': timestamp,
            'values': values,
            'prediction': str(prediction)
        }
        
        # Get existing predictions
        predictions = []
        if self.store.exists('predictions'):
            predictions = self.store.get('predictions')['data']
        
        # Add new prediction
        predictions.append(prediction_data)
        
        # Keep only last 50 predictions
        if len(predictions) > 50:
            predictions = predictions[-50:]
        
        # Save to storage
        self.store.put('predictions', data=predictions)
    
    def show_history(self, instance):
        """Show prediction history"""
        if not self.store.exists('predictions'):
            self.show_info('No prediction history found')
            return
        
        predictions = self.store.get('predictions')['data']
        
        # Create history popup
        content = BoxLayout(orientation='vertical', padding=10, spacing=10)
        
        # Title
        title = Label(
            text=f'Prediction History ({len(predictions)} entries)',
            size_hint_y=None,
            height=40,
            bold=True
        )
        content.add_widget(title)
        
        # Create scrollable history
        history_layout = GridLayout(cols=1, spacing=5, size_hint_y=None)
        history_layout.bind(minimum_height=history_layout.setter('height'))
        
        for pred in reversed(predictions[-20:]):  # Show last 20
            # Format timestamp
            timestamp = datetime.datetime.fromisoformat(pred['timestamp'])
            formatted_time = timestamp.strftime('%Y-%m-%d %H:%M')
            
            # Create history item
            item_text = f'[{formatted_time}] Class {pred["prediction"]}\n'
            item_text += f'Values: {", ".join([f"{v:.1f}" for v in pred["values"][:3]])}...'
            
            item = Label(
                text=item_text,
                size_hint_y=None,
                height=60,
                text_size=(Window.width - 40, None),
                halign='left',
                valign='top',
                color=(0.3, 0.3, 0.3, 1)
            )
            history_layout.add_widget(item)
        
        # Scroll view
        scroll_view = ScrollView(size_hint=(1, 1))
        scroll_view.add_widget(history_layout)
        content.add_widget(scroll_view)
        
        # Close button
        close_btn = Button(
            text='Close',
            size_hint_y=None,
            height=40,
            background_color=(0.8, 0.2, 0.2, 1)
        )
        content.add_widget(close_btn)
        
        # Create popup
        popup = Popup(
            title='History',
            content=content,
            size_hint=(0.9, 0.8)
        )
        
        close_btn.bind(on_press=popup.dismiss)
        popup.open()
    
    def show_error(self, message):
        """Show error popup"""
        content = BoxLayout(orientation='vertical', padding=10)
        content.add_widget(Label(text=message, color=(0.8, 0.2, 0.2, 1)))
        
        close_btn = Button(
            text='OK',
            size_hint_y=None,
            height=40,
            background_color=(0.8, 0.2, 0.2, 1)
        )
        content.add_widget(close_btn)
        
        popup = Popup(
            title='Error',
            content=content,
            size_hint=(0.8, 0.4)
        )
        
        close_btn.bind(on_press=popup.dismiss)
        popup.open()
    
    def show_info(self, message):
        """Show info popup"""
        content = BoxLayout(orientation='vertical', padding=10)
        content.add_widget(Label(text=message, color=(0.2, 0.6, 1, 1)))
        
        close_btn = Button(
            text='OK',
            size_hint_y=None,
            height=40,
            background_color=(0.2, 0.6, 1, 1)
        )
        content.add_widget(close_btn)
        
        popup = Popup(
            title='Info',
            content=content,
            size_hint=(0.8, 0.4)
        )
        
        close_btn.bind(on_press=popup.dismiss)
        popup.open()

if __name__ == '__main__':
    DiabetesPredictionApp().run() 