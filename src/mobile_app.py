import os
import joblib
import pandas as pd
import datetime
from kivy.metrics import dp
from kivy.storage.jsonstore import JsonStore
from kivy.core.window import Window
from kivy.clock import Clock
from kivy.lang import Builder
from kivymd.app import MDApp
from kivymd.uix.screen import MDScreen
from kivymd.uix.button import MDFabButton, MDButton
from kivymd.uix.textfield import MDTextField
from kivymd.uix.card import MDCard
from kivymd.uix.label import MDLabel
from kivymd.uix.dialog import MDDialog
from kivymd.uix.chip import MDChip
from kivymd.uix.snackbar import MDSnackbar
from kivymd.uix.list import MDListItem
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.scrollview import MDScrollView
from kivymd.uix.divider import MDDivider
from kivymd.uix.gridlayout import MDGridLayout

# Clean & Organized KV Design
KV = '''
MDScreen:
    md_bg_color: 0.95, 0.97, 1, 1
    
    # Clean Background
    canvas.before:
        Color:
            rgba: 0.95, 0.97, 1, 1
        Rectangle:
            pos: self.pos
            size: self.size
    
    # Clean Header
    MDBoxLayout:
        orientation: "vertical"
        size_hint_y: None
        height: dp(120)
        md_bg_color: 0.2, 0.6, 0.8, 1
        padding: dp(16)
        elevation: 8
        
        # App Title
        MDLabel:
            text: "🏥 Diabetes Health Buddy"
            theme_text_color: "Custom"
            text_color: 1, 1, 1, 1
            halign: "center"
            bold: True
            font_size: dp(24)
            size_hint_y: None
            height: dp(40)
            
        # Subtitle
        MDLabel:
            text: "Your Health Adventure Starts Here!"
            theme_text_color: "Custom"
            text_color: 1, 1, 1, 0.9
            halign: "center"
            font_size: dp(16)
            size_hint_y: None
            height: dp(30)
            
        # Clean Divider
        MDDivider:
            md_bg_color: 1, 1, 1, 0.3
            height: dp(2)
    
    # Main Content
    MDScrollView:
        pos_hint: {"top": 1}
        size_hint_y: None
        height: self.parent.height - dp(220)
        
        MDBoxLayout:
            orientation: "vertical"
            padding: dp(16)
            spacing: dp(16)
            size_hint_y: None
            height: self.minimum_height
            
            # Welcome Card
            MDCard:
                orientation: "vertical"
                size_hint_y: None
                height: dp(80)
                padding: dp(16)
                elevation: 4
                radius: [12]
                md_bg_color: 1, 0.9, 0.7, 1
                
                MDLabel:
                    text: "🎉 Welcome to Your Health Journey!"
                    theme_text_color: "Custom"
                    text_color: 0.6, 0.4, 0.2, 1
                    halign: "center"
                    bold: True
                    font_size: dp(18)
                    size_hint_y: None
                    height: dp(30)
                    
                MDLabel:
                    text: "Enter your health numbers below"
                    theme_text_color: "Custom"
                    text_color: 0.6, 0.4, 0.2, 1
                    halign: "center"
                    font_size: dp(14)
                    size_hint_y: None
                    height: dp(30)
            
            # Input Form Section
            MDCard:
                orientation: "vertical"
                size_hint_y: None
                height: self.minimum_height
                padding: dp(16)
                elevation: 6
                radius: [12]
                md_bg_color: 1, 1, 1, 1
                
                MDLabel:
                    text: "📊 Health Parameters"
                    theme_text_color: "Custom"
                    text_color: 0.2, 0.4, 0.6, 1
                    halign: "center"
                    bold: True
                    font_size: dp(18)
                    size_hint_y: None
                    height: dp(30)
                
                MDDivider:
                    md_bg_color: 0.2, 0.6, 0.8, 0.5
                    height: dp(2)
                
                MDGridLayout:
                    id: form_grid
                    cols: 1
                    adaptive_height: True
                    spacing: dp(12)
                
                MDDivider:
                    md_bg_color: 0.2, 0.6, 0.8, 0.5
                    height: dp(2)
                
                # Results Section
                MDBoxLayout:
                    id: results_box
                    orientation: "vertical"
                    adaptive_height: True
                    spacing: dp(12)
                    padding: dp(12)
                    md_bg_color: 0.95, 0.97, 1, 0.8
                    radius: [8]
                
                MDDivider:
                    md_bg_color: 0.2, 0.6, 0.8, 0.5
                    height: dp(2)
                
                # Progress Section
                MDBoxLayout:
                    id: progress_box
                    orientation: "vertical"
                    size_hint_y: None
                    height: dp(40)
                    opacity: 0
                    spacing: dp(8)
                    
                    MDLabel:
                        text: "🤖 AI is analyzing your data..."
                        theme_text_color: "Custom"
                        text_color: 0.2, 0.6, 0.8, 1
                        halign: "center"
                        font_size: dp(14)
                        size_hint_y: None
                        height: dp(20)
                    
                    ProgressBar:
                        id: progress
                        max: 100
                        value: 0
                        size_hint_y: None
                        height: dp(8)
                        color: 0.2, 0.6, 0.8, 1
    
    # Action Buttons
    MDBoxLayout:
        orientation: "horizontal"
        size_hint_y: None
        height: dp(100)
        spacing: dp(16)
        padding: dp(16)
        pos_hint: {"bottom": 1}
        
        # Predict Button
        MDFabButton:
            icon: "heart-pulse"
            md_bg_color: 0.2, 0.8, 0.4, 1
            on_release: app.on_predict()
            tooltip_text: "Get Health Assessment"
            size_hint: None, None
            size: dp(60), dp(60)
            pos_hint: {"center_x": 0.5}
            elevation: 6
            radius: [30]
        
        # History Button
        MDFabButton:
            icon: "chart-line"
            md_bg_color: 0.8, 0.4, 0.2, 1
            on_release: app.on_history()
            tooltip_text: "View History"
            size_hint: None, None
            size: dp(60), dp(60)
            pos_hint: {"center_x": 0.5}
            elevation: 6
            radius: [30]
        
        # Info Button
        MDFabButton:
            icon: "help-circle"
            md_bg_color: 0.4, 0.6, 0.8, 1
            on_release: app.show_info("This app helps you check your health status using AI. Enter your medical numbers and get instant results!")
            tooltip_text: "App Help"
            size_hint: None, None
            size: dp(60), dp(60)
            pos_hint: {"center_x": 0.5}
            elevation: 6
            radius: [30]
'''

class DiabetesApp(MDApp):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.features = ['Gender', 'AGE', 'Urea', 'Cr', 'HbA1c', 'Chol', 'TG', 'HDL', 'LDL', 'VLDL', 'BMI']
        self.feature_ranges = {
            'Gender': (0, 1), 'AGE': (18, 100), 'Urea': (1.0, 50.0), 'Cr': (5, 1000),
            'HbA1c': (3.0, 15.0), 'Chol': (1.0, 10.0), 'TG': (0.1, 50.0), 'HDL': (0.1, 5.0),
            'LDL': (0.1, 10.0), 'VLDL': (0.1, 50.0), 'BMI': (15.0, 50.0)
        }
        self.inputs = {}
        self.model = None
        self.store = JsonStore('diabetes_predictions.json')

    def build(self):
        self.theme_cls.primary_palette = "Blue"
        self.theme_cls.primary_hue = "600"
        self.theme_cls.theme_style = "Light"
        Window.size = (400, 800)
        self.screen = Builder.load_string(KV)
        self.load_model()
        self.build_form()
        self.show_welcome()
        return self.screen

    def load_model(self):
        model_paths = ['diabetes_model.pkl', 'src/diabetes_model.pkl', '../diabetes_model.pkl']
        for path in model_paths:
            if os.path.exists(path):
                self.model = joblib.load(path)
                break
        if not self.model:
            self.show_error("Model not found!")

    def build_form(self):
        grid = self.screen.ids.form_grid
        grid.clear_widgets()
        
        # Clean feature styling
        feature_styles = {
            'Gender': {'emoji': '👤', 'color': (0.6, 0.2, 0.8, 1), 'desc': 'Gender (0=Female, 1=Male)'},
            'AGE': {'emoji': '📅', 'color': (0.2, 0.6, 0.2, 1), 'desc': 'Age in years'},
            'Urea': {'emoji': '🧪', 'color': (0.8, 0.4, 0.2, 1), 'desc': 'Blood urea nitrogen'},
            'Cr': {'emoji': '💊', 'color': (0.4, 0.2, 0.8, 1), 'desc': 'Creatinine level'},
            'HbA1c': {'emoji': '🩸', 'color': (0.8, 0.2, 0.2, 1), 'desc': 'Glycated hemoglobin'},
            'Chol': {'emoji': '🫀', 'color': (0.2, 0.6, 0.8, 1), 'desc': 'Total cholesterol'},
            'TG': {'emoji': '🩺', 'color': (0.8, 0.6, 0.2, 1), 'desc': 'Triglycerides'},
            'HDL': {'emoji': '❤️', 'color': (0.8, 0.2, 0.6, 1), 'desc': 'High-density lipoprotein'},
            'LDL': {'emoji': '💙', 'color': (0.2, 0.4, 0.8, 1), 'desc': 'Low-density lipoprotein'},
            'VLDL': {'emoji': '💜', 'color': (0.6, 0.2, 0.6, 1), 'desc': 'Very low-density lipoprotein'},
            'BMI': {'emoji': '⚖️', 'color': (0.6, 0.4, 0.4, 1), 'desc': 'Body mass index'}
        }
        
        for feature in self.features:
            min_val, max_val = self.feature_ranges[feature]
            style = feature_styles.get(feature, {'emoji': '📊', 'color': (0.4, 0.4, 0.4, 1), 'desc': feature})
            
            # Create clean input card
            card = MDCard(
                orientation="vertical",
                size_hint_y=None,
                height=dp(80),
                padding=dp(12),
                elevation=2,
                radius=[8],
                md_bg_color=(0.98, 0.98, 0.98, 1),
                spacing=dp(6)
            )
            
            # Clean header
            header = MDBoxLayout(
                orientation="horizontal",
                size_hint_y=None,
                height=dp(25),
                spacing=dp(8)
            )
            
            emoji_label = MDLabel(
                text=style['emoji'],
                theme_text_color="Custom",
                text_color=style['color'],
                font_size=dp(16),
                size_hint_x=None,
                width=dp(25)
            )
            
            name_label = MDLabel(
                text=feature,
                theme_text_color="Custom",
                text_color=style['color'],
                bold=True,
                font_size=dp(14),
                size_hint_x=0.6
            )
            
            range_label = MDLabel(
                text=f"({min_val}-{max_val})",
                theme_text_color="Custom",
                text_color=style['color'][:3] + (0.6,),
                font_size=dp(12),
                size_hint_x=0.4,
                halign="right"
            )
            
            header.add_widget(emoji_label)
            header.add_widget(name_label)
            header.add_widget(range_label)
            
            # Clean input field
            ti = MDTextField(
                hint_text=f"Enter {feature}",
                mode="outlined",
                size_hint_y=None,
                height=dp(40),
                input_filter="float",
                font_size=dp(14)
            )
            
            # Clean description
            desc_label = MDLabel(
                text=style['desc'],
                theme_text_color="Custom",
                text_color=style['color'][:3] + (0.6,),
                font_size=dp(11),
                size_hint_y=None,
                height=dp(15)
            )
            
            card.add_widget(header)
            card.add_widget(ti)
            card.add_widget(desc_label)
            
            self.inputs[feature] = ti
            grid.add_widget(card)

    def show_welcome(self):
        # Skip welcome dialog for now
        pass

    def get_health_status(self, prediction):
        status_map = {
            0: {'status': 'Normal/Healthy', 'color': (0.2, 0.8, 0.2, 1), 'desc': 'Your health markers are within normal ranges.', 'risk': 'Low Risk', 'insights': 'Keep maintaining your healthy lifestyle!'},
            1: {'status': 'Prediabetic', 'color': (1.0, 0.7, 0.1, 1), 'desc': 'Your markers suggest prediabetes. Early intervention is recommended.', 'risk': 'Moderate Risk', 'insights': 'Lifestyle changes and monitoring are advised.'},
            2: {'status': 'Diabetic', 'color': (0.9, 0.2, 0.2, 1), 'desc': 'Your markers indicate diabetes. Medical consultation is strongly advised.', 'risk': 'High Risk', 'insights': 'Immediate medical attention recommended.'}
        }
        return status_map.get(prediction, status_map[0])

    def on_predict(self):
        progress_box = self.screen.ids.progress_box
        progress = self.screen.ids.progress
        progress_box.opacity = 1
        progress.value = 0
        
        values = []
        for feature in self.features:
            value = self.inputs[feature].text.strip()
            if not value:
                self.show_error(f'Please enter {feature}')
                progress_box.opacity = 0
                return
            try:
                values.append(float(value))
            except ValueError:
                self.show_error(f'{feature} must be a number')
                progress_box.opacity = 0
                return
        
        progress.value = 30
        errors = self.validate_input(values)
        if errors:
            error_msg = 'Validation Errors:\n' + '\n'.join(errors[:3])
            if len(errors) > 3:
                error_msg += f'\n... and {len(errors) - 3} more errors'
            self.show_error(error_msg)
            progress_box.opacity = 0
            return
        
        progress.value = 60
        values_df = pd.DataFrame([values], columns=self.features)
        prediction = self.model.predict(values_df)[0]
        progress.value = 100
        health_info = self.get_health_status(prediction)
        self.show_results(health_info, prediction)
        self.save_prediction(values, prediction, health_info)
        progress_box.opacity = 0

    def show_results(self, health_info, prediction):
        results_box = self.screen.ids.results_box
        results_box.clear_widgets()
        
        # Clean status card
        status_card = MDCard(
            orientation="vertical",
            size_hint_y=None,
            height=dp(120),
            padding=dp(16),
            elevation=4,
            radius=[8],
            md_bg_color=health_info['color'][:3] + (0.1,),
            spacing=dp(8)
        )
        
        # Status header
        status_header = MDBoxLayout(
            orientation="horizontal",
            size_hint_y=None,
            height=dp(30),
            spacing=dp(12)
        )
        
        status_icon = MDLabel(
            text="✅" if prediction == 0 else "⚠️" if prediction == 1 else "🚨",
            theme_text_color="Custom",
            text_color=health_info['color'],
            font_size=dp(20),
            size_hint_x=None,
            width=dp(30)
        )
        
        status_label = MDLabel(
            text=health_info['status'],
            theme_text_color="Custom",
            text_color=health_info['color'],
            bold=True,
            font_size=dp(16),
            size_hint_x=0.7
        )
        
        status_header.add_widget(status_icon)
        status_header.add_widget(status_label)
        status_card.add_widget(status_header)
        
        # Risk level
        risk_card = MDCard(
            orientation="horizontal",
            size_hint_y=None,
            height=dp(30),
            padding=dp(8),
            elevation=2,
            radius=[6],
            md_bg_color=health_info['color'][:3] + (0.2,),
            spacing=dp(8)
        )
        
        risk_icon = MDLabel(
            text="🎯",
            theme_text_color="Custom",
            text_color=health_info['color'],
            font_size=dp(14),
            size_hint_x=None,
            width=dp(20)
        )
        
        risk_label = MDLabel(
            text=f"Risk: {health_info['risk']}",
            theme_text_color="Custom",
            text_color=health_info['color'],
            bold=True,
            font_size=dp(14)
        )
        
        risk_card.add_widget(risk_icon)
        risk_card.add_widget(risk_label)
        status_card.add_widget(risk_card)
        
        # AI classification
        ai_card = MDCard(
            orientation="horizontal",
            size_hint_y=None,
            height=dp(30),
            padding=dp(8),
            elevation=2,
            radius=[6],
            md_bg_color=(0.2, 0.6, 0.8, 0.2),
            spacing=dp(8)
        )
        
        ai_icon = MDLabel(
            text="🤖",
            theme_text_color="Custom",
            text_color=(0.2, 0.6, 0.8, 1),
            font_size=dp(14),
            size_hint_x=None,
            width=dp(20)
        )
        
        ai_label = MDLabel(
            text=f"AI Class: {prediction}",
            theme_text_color="Custom",
            text_color=(0.2, 0.6, 0.8, 1),
            font_size=dp(14)
        )
        
        ai_card.add_widget(ai_icon)
        ai_card.add_widget(ai_label)
        status_card.add_widget(ai_card)
        
        results_box.add_widget(status_card)
        
        # Insights card
        insights_card = MDCard(
            orientation="vertical",
            size_hint_y=None,
            height=dp(80),
            padding=dp(12),
            elevation=4,
            radius=[8],
            md_bg_color=(0.95, 0.97, 1, 0.9),
            spacing=dp(8)
        )
        
        insights_header = MDLabel(
            text="💡 Health Insights",
            theme_text_color="Custom",
            text_color=(0.2, 0.4, 0.6, 1),
            bold=True,
            font_size=dp(14),
            size_hint_y=None,
            height=dp(20)
        )
        
        insights_text = MDLabel(
            text=health_info['insights'],
            theme_text_color="Custom",
            text_color=(0.2, 0.4, 0.6, 1),
            font_size=dp(12),
            size_hint_y=None,
            height=dp(40)
        )
        
        insights_card.add_widget(insights_header)
        insights_card.add_widget(insights_text)
        results_box.add_widget(insights_card)
        
        # Description card
        desc_card = MDCard(
            orientation="vertical",
            size_hint_y=None,
            height=dp(60),
            padding=dp(12),
            elevation=4,
            radius=[8],
            md_bg_color=(1, 0.95, 0.9, 0.9),
            spacing=dp(8)
        )
        
        desc_header = MDLabel(
            text="📋 Medical Description",
            theme_text_color="Custom",
            text_color=(0.6, 0.4, 0.2, 1),
            bold=True,
            font_size=dp(14),
            size_hint_y=None,
            height=dp(20)
        )
        
        desc_text = MDLabel(
            text=health_info['desc'],
            theme_text_color="Custom",
            text_color=(0.6, 0.4, 0.2, 1),
            font_size=dp(12),
            size_hint_y=None,
            height=dp(30)
        )
        
        desc_card.add_widget(desc_header)
        desc_card.add_widget(desc_text)
        results_box.add_widget(desc_card)

    def validate_input(self, values):
        errors = []
        for i, (feature, value) in enumerate(zip(self.features, values)):
            min_val, max_val = self.feature_ranges[feature]
            if value < min_val or value > max_val:
                errors.append(f"{feature}: {value} (should be between {min_val} and {max_val})")
        return errors

    def save_prediction(self, values, prediction, health_info):
        timestamp = datetime.datetime.now().isoformat()
        prediction_data = {
            'timestamp': timestamp,
            'values': values,
            'prediction': str(prediction),
            'status': health_info['status'],
            'risk': health_info['risk']
        }
        predictions = []
        if self.store.exists('predictions'):
            predictions = self.store.get('predictions')['data']
        predictions.append(prediction_data)
        if len(predictions) > 50:
            predictions = predictions[-50:]
        self.store.put('predictions', data=predictions)

    def on_history(self):
        if not self.store.exists('predictions'):
            self.show_info('No prediction history found')
            return
        
        predictions = self.store.get('predictions')['data']
        history_text = "Prediction History:\n\n"
        
        for pred in reversed(predictions[-10:]):
            timestamp = datetime.datetime.fromisoformat(pred['timestamp'])
            formatted_time = timestamp.strftime('%Y-%m-%d %H:%M')
            status = pred.get("status", f"Class {pred['prediction']}")
            history_text += f"[{formatted_time}] {status}\n"
            history_text += f"Risk: {pred.get('risk', 'Unknown')}\n"
            history_text += f"Values: {', '.join([f'{v:.1f}' for v in pred['values'][:3]])}...\n\n"
        
        self.show_info(history_text[:150] + "..." if len(history_text) > 150 else history_text)

    def show_error(self, message):
        MDSnackbar(
            text=f"❌ {message}",
            bg_color=(0.8, 0.2, 0.2, 1),
            duration=4
        ).open()

    def show_info(self, message):
        MDSnackbar(
            text=f"ℹ️ {message}",
            bg_color=(0.2, 0.6, 1, 1),
            duration=5
        ).open()

if __name__ == '__main__':
    DiabetesApp().run() 