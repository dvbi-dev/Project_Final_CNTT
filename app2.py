import base64
import io
import os
import random
from pathlib import Path
import qrcode
from io import BytesIO
from flask import send_file
import cv2
from PIL import Image as PILImage
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash
from sqlalchemy import ForeignKey, func, distinct, or_, and_
from sqlalchemy.orm import relationship
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from flask_bcrypt import Bcrypt
from werkzeug.utils import secure_filename
from sqlalchemy import not_
import neuralStyleProcess
from filter import cartoonize_image, Summer, sepia, Winter, invert, HDR, sharpen, pencil_sketch_grey, pencil_sketch_col, \
    tv60, apply_emboss_effect, adjust_lightness, pixelize_image
from blackwhitecolor import net
from feature_extractor import FeatureExtractor
from imagecap import classify
from flask_mail import Mail, Message
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
fe = FeatureExtractor()
source_path = Path("./static/uploads")
destination_path = Path("./static/feature")
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:@127.0.0.1:3306/test11'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'static/uploads'  # Folder where uploaded images will be stored
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False
app.config['MAIL_USERNAME'] = 'biduongnote1@gmail.com' # Your Gmail email address
app.config['MAIL_PASSWORD'] ='wxit rxbo urkd vylp'  # Your Gmail password
app.config['MAIL_DEFAULT_SENDER'] = 'dvbi.19it5@vku.udn.vn'  # Your default sender email address
app.secret_key = 'biduong'
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
features = []
img_paths = []

for feature_path in Path("./static/feature").glob("*.npy"):
    features.append(np.load(feature_path))
    img_paths.append(Path("./static/uploads") / (feature_path.stem + ".jpg"))
features = np.array(features)
class User(db.Model):
    __tablename__ = 'user'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    avatar = db.Column(db.String(200), nullable=True)
    followers = relationship('Follow', foreign_keys='Follow.following_id', backref='following', lazy='dynamic')
    following = relationship('Follow', foreign_keys='Follow.follower_id', backref='follower', lazy='dynamic')
    images = relationship('Image', backref='user', lazy=True)
    comments = relationship('Comment', backref='user', lazy=True)
    likes = relationship('Like', backref='user', lazy=True)
    albums = relationship('Album', backref='user', lazy=True)
    type = db.Column(db.Integer, nullable=True)
    password_reset_code = db.Column(db.String(6), nullable=True)
    # ... (other fields)

    def follow(self, user_to_follow):
        if not self.is_following(user_to_follow):
            follow = Follow(follower=self, following=user_to_follow)
            db.session.add(follow)
            db.session.commit()

    def unfollow(self, user_to_unfollow):
        follow = self.following.filter_by(following_id=user_to_unfollow.id).first()
        if follow:
            db.session.delete(follow)
            db.session.commit()

    def is_following(self, user):
        return self.following.filter_by(following_id=user.id).first() is not None

    def like_image(self, image):
        if not self.has_liked(image):
            like = Like(user=self, image=image)
            db.session.add(like)
            db.session.commit()

    def unlike(self, image):
        if self.has_liked(image):
            like = Like.query.filter_by(user_id=self.id, image_id=image.id).first()
            if like:
                try:
                    db.session.delete(like)
                    db.session.commit()
                except Exception as e:
                    print(f"Error during unlike: {e}")
            else:
                print("Like not found")
        else:
            print("User has not liked the image")

    def has_liked(self, image):
        return any(like.image == image for like in self.likes)

    # Add this function to your Flask application
    def has_user_liked_image(user_id, likes):
        return any(like.user_id == user_id for like in likes)

    def save_image(self, image):
        if not self.has_saved(image):
            saved_image = SavedImage(user=self, image=image)
            db.session.add(saved_image)
            db.session.commit()

    def unsave_image(self, image):
        saved_image = SavedImage.query.filter_by(user_id=self.id, image_id=image.id).first()
        if saved_image:
            db.session.delete(saved_image)
            db.session.commit()

    def has_saved(self, image):
        return any(saved.image_id == image.id for saved in self.saved_images)

    def suggested_users(self, limit=10):
        # # Subquery to get the IDs of users that the current user is following
        # following_subquery = db.session.query(Follow.following_id) \
        #     .filter(Follow.follower_id == self.id) \
        #     .subquery()
        #
        # # Query to get users who are being followed by users followed by the current user
        # suggested_users = db.session.query(User) \
        #     .join(Follow, User.id == Follow.follower_id) \
        #     .filter(Follow.following_id.in_(following_subquery)) \
        #     .filter(Follow.follower_id != self.id) \
        #     .group_by(User.id) \
        #     .order_by(func.random()) \
        #     .limit(limit) \
        #     .all()
        following_subquery = db.session.query(Follow.following_id) \
            .filter(Follow.follower_id == self.id) \
            .subquery()

        # Subquery to get the IDs of users that the friends of the current user are following
        friends_following_subquery = db.session.query(Follow.following_id) \
            .filter(Follow.follower_id.in_(following_subquery)) \
            .filter(Follow.follower_id != self.id) \
            .subquery()

        # Query to get users who are being followed by friends of the current user
        suggested_users = db.session.query(User) \
            .join(Follow, User.id == Follow.follower_id) \
            .filter(Follow.following_id.in_(friends_following_subquery)) \
            .filter(not_(User.id.in_(following_subquery))) \
            .group_by(User.id) \
            .order_by(func.random()) \
            .limit(limit) \
            .all()
        return suggested_users

    def collaborative_filtering_recommendation(self, limit=10):
        # Lấy danh sách người dùng và mục (các người dùng mà họ đã theo dõi)
        users = User.query.all()
        items = Image.query.all()

        # Tạo ma trận user-item
        user_item_matrix = [[0] * len(items) for _ in range(len(users))]

        for i, user in enumerate(users):
            for j, item in enumerate(items):
                user_item_matrix[i][j] = int(user.has_liked(item))

        # Tính độ tương đồng cosine giữa các người dùng
        user_similarity_matrix = cosine_similarity(user_item_matrix)

        # Lấy người dùng giống nhất với người dùng hiện tại
        similar_users_index = user_similarity_matrix[self.id - 1].argsort()[-2::-1]

        # Lấy danh sách mục mà người dùng giống nhất đã thích nhưng người dùng hiện tại chưa thích
        recommended_items = []
        for j in range(len(items)):
            if user_item_matrix[self.id - 1][j] == 0 and any(user_item_matrix[i][j] == 1 for i in similar_users_index):
                recommended_items.append(items[j])
                if len(recommended_items) == limit:
                    break

        return recommended_items

    def collaborative_filtering_user_recommendation(self, limit=10):
        # Lấy danh sách người dùng và mục (các người dùng mà họ đã theo dõi)
        users = User.query.all()
        images = Image.query.all()

        current_user_features = [self.likes.count(), self.followers.count()]

        # Tính toán sự tương đồng cosine giữa người dùng hiện tại và tất cả người dùng khác
        user_similarity_scores = [
            cosine_similarity(current_user_features, [user.likes.count(), user.followers.count()])
            for user in users
        ]

        # Lấy danh sách người dùng giống nhất với người dùng hiện tại
        similar_users_index = sorted(range(len(user_similarity_scores)), key=lambda k: user_similarity_scores[k],
                                     reverse=True)

        # Lấy danh sách người dùng mà người dùng giống nhất đã thích nhưng người dùng hiện tại chưa thích
        suggested_users = [
                              users[i]
                              for i in similar_users_index
                              if not self.has_user_liked_image(users[i].id, images[i].likes)
                          ][:limit]

        return suggested_users

    def de_xuat_hashtags(self, so_luong_de_xuat=5):
        # Lấy danh sách các hashtag của người dùng
        anh_da_like = db.session.query(Image). \
            join(Like, Like.image_id == Image.id). \
            join(User, User.id == Like.user_id). \
            filter(User.id == self.id).all()

        # Lấy các hashtag từ những ảnh mà người dùng đã like
        hashtags_da_like = []
        for anh in anh_da_like:
            hashtags_da_like += db.session.query(Hashtag.name). \
                join(ImageHashtags, Hashtag.id == ImageHashtags.hashtag_id). \
                filter(ImageHashtags.image_id == anh.id).all()

        hashtags_da_like = [tag[0] for tag in hashtags_da_like]

        # Lấy tất cả các hashtag trong hệ thống
        tat_ca_hashtag = db.session.query(Hashtag).all()
        tat_ca_hashtag = [tag.name for tag in tat_ca_hashtag]

        # Tạo ma trận tần suất hashtag từ các ảnh đã like
        matrix_tan_suat = np.zeros((1, len(tat_ca_hashtag)))
        for hashtag in hashtags_da_like:
            index = tat_ca_hashtag.index(hashtag)
            matrix_tan_suat[0, index] += 1

        # Tính tương đồng cosine giữa người dùng và tất cả các người dùng khác
        if len(matrix_tan_suat.nonzero()[1]) == 0:
            return []  # Trả về danh sách rỗng nếu người dùng không like ảnh nào

        tinh_tuong_dong = cosine_similarity(matrix_tan_suat, features[:, :len(tat_ca_hashtag)])[0]

        # Lấy danh sách các người dùng có tương đồng cao nhất
        nguoi_dung_goi_y = np.argsort(tinh_tuong_dong)[::-1][1:so_luong_de_xuat + 1]

        # Lấy các hashtag từ những người dùng được đề xuất
        hashtags_de_xuat = []
        for user_index in nguoi_dung_goi_y:
            hashtags_de_xuat += db.session.query(Hashtag.name). \
                join(ImageHashtags, Hashtag.id == ImageHashtags.hashtag_id). \
                join(Image, Image.id == ImageHashtags.image_id). \
                join(User, User.id == Image.user_id). \
                filter(User.id == user_index + 1).all()

        hashtags_de_xuat = [tag[0] for tag in hashtags_de_xuat]

        # Lọc bỏ các hashtag đã sử dụng bởi người dùng hiện tại
        hashtags_de_xuat = [tag for tag in hashtags_de_xuat if tag not in hashtags_da_like]


        return hashtags_de_xuat[:so_luong_de_xuat]


    def send_message(self, receiver, content):
        message = ChatMessage(sender_id=self.id, receiver_id=receiver.id, content=content)
        db.session.add(message)
        db.session.commit()

    def get_messages(self, partner_id):
        # Lấy tất cả tin nhắn giữa người dùng hiện tại và đối tác
        messages = ChatMessage.query.filter(
            or_(
                and_(ChatMessage.sender_id == self.id, ChatMessage.receiver_id == partner_id),
                and_(ChatMessage.sender_id == partner_id, ChatMessage.receiver_id == self.id)
            )
        ).order_by(ChatMessage.timestamp).all()
        return messages

    def has_viewed_enough_images_in_pinboard(self, pinboard_id, min_view_count=2):
        viewed_images_count = db.session.query(func.count(distinct(ViewHistory.image_id))) \
            .filter(ViewHistory.user_id == self.id, ViewHistory.image_id.in_(db.session.query(PinnedImage.image_id)
                                                                             .filter(
            PinnedImage.pin_board_id == pinboard_id))) \
            .scalar()
        return viewed_images_count >= min_view_count

    def suggested_pin_boards(self, limit=5):
        # Lấy danh sách các Pin Board
        pin_boards = PinBoard.query.all()

        # Lọc Pin Board dựa trên điều kiện đã xem ít nhất 3 hình ảnh trong đó
        suggested_pin_boards = [pin_board for pin_board in pin_boards
                                if self.has_viewed_enough_images_in_pinboard(pin_board.id)]

        return suggested_pin_boards[:limit]

    # ... (other fields)
class ReportImage(db.Model):
    __tablename__ = 'report_image'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    image_id = db.Column(db.Integer, db.ForeignKey('image.id'), nullable=False)
    reason = db.Column(db.String(255), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    # Define relationships
    user = relationship('User', backref='reported_images', foreign_keys=[user_id], lazy=True)
    image = relationship('Image', backref='reports', foreign_keys=[image_id], lazy=True)

class ViewHistory(db.Model):
    __tablename__ = 'view_history'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, ForeignKey('user.id'), nullable=False)
    image_id = db.Column(db.Integer, ForeignKey('image.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    # Define relationships
    user = relationship('User', backref='view_history', lazy=True)
    image = relationship('Image', backref='view_history', lazy=True)
class SavedImage(db.Model):
    __tablename__ = 'savedimage'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, ForeignKey('user.id'), nullable=False)
    image_id = db.Column(db.Integer, ForeignKey('image.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    # Define relationships
    user = relationship('User', backref='saved_images', lazy=True)
    image = relationship('Image', backref='saved_images', lazy=True)
class Album(db.Model):
    __tablename__ = 'album'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, ForeignKey('user.id'), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    images = relationship('Image', backref='album', lazy=True)
class Image(db.Model):
    __tablename__ = 'image'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, ForeignKey('user.id'), nullable=False)
    image_path = db.Column(db.String(200), nullable=False)
    caption = db.Column(db.String(200), nullable=True)
    description = db.Column(db.String(255), nullable=True)  # Thêm cột mô tả
    link = db.Column(db.String(255), nullable=True)  # Thêm cột liên kết
    views = db.Column(db.Integer, default=0)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    comments = relationship('Comment', backref='image', lazy=True)
    allow_comments = db.Column(db.Boolean, default=True, nullable=False)
    likes = relationship('Like', backref='image', lazy=True)
    album_id = db.Column(db.Integer, db.ForeignKey('album.id'), nullable=True)
    hashtags = relationship('Hashtag', secondary='image_hashtags', backref='images', lazy=True)


class Comment(db.Model):
    __tablename__ = 'comment'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, ForeignKey('user.id'), nullable=False)
    image_id = db.Column(db.Integer, ForeignKey('image.id'), nullable=False)
    comment_text = db.Column(db.String(200), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
class Like(db.Model):
    __tablename__ = 'like'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    image_id = db.Column(db.Integer, db.ForeignKey('image.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
class Follow(db.Model):
    __tablename__ = 'follow'
    id = db.Column(db.Integer, primary_key=True)
    follower_id = db.Column(db.Integer, ForeignKey('user.id'), nullable=False)
    following_id = db.Column(db.Integer, ForeignKey('user.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class Hashtag(db.Model):
    __tablename__ = 'hashtag'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), unique=True, nullable=False)

class ImageHashtags(db.Model):
    __tablename__ = 'image_hashtags'
    id = db.Column(db.Integer, primary_key=True)
    image_id = db.Column(db.Integer, ForeignKey('image.id'), nullable=False)
    hashtag_id = db.Column(db.Integer, ForeignKey('hashtag.id'), nullable=False)

class PinBoard(db.Model):
    __tablename__ = 'pin_board'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    pinned_images = db.relationship('PinnedImage', backref='pin_board', lazy=True)

class PinnedImage(db.Model):
    __tablename__ = 'pinned_image'
    id = db.Column(db.Integer, primary_key=True)
    image_id = db.Column(db.Integer, db.ForeignKey('image.id'), nullable=False)
    pin_board_id = db.Column(db.Integer, db.ForeignKey('pin_board.id'), nullable=False)

class ChatMessage(db.Model):
    __tablename__ = 'chat_message'
    id = db.Column(db.Integer, primary_key=True)
    sender_id = db.Column(db.Integer, ForeignKey('user.id'), nullable=False)
    receiver_id = db.Column(db.Integer, ForeignKey('user.id'), nullable=False)
    content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    sender = db.relationship('User', foreign_keys=[sender_id], backref='sent_messages', lazy=True)
    receiver = db.relationship('User', foreign_keys=[receiver_id], backref='received_messages', lazy=True)


# Route for user registration
def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif','jfit'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
mail = Mail(app)
def send_blocked_email(recipient_email):
    # Create a message
    subject = 'Account Blocked'
    body = 'Your account has been blocked. Please contact support for more information.'

    message = Message(subject=subject, recipients=[recipient_email], body=body)

    # Send the message
    mail.send(message)

def send_unblocked_email(recipient_email):
    # Create a message
    subject = 'Account Unblocked'
    body = 'Your account has been unblocked. Thank you for your understanding.'

    message = Message(subject=subject, recipients=[recipient_email], body=body)

    # Send the message
    mail.send(message)

def recommend_userss(user_id):
    # Tái tính toán danh sách đề xuất (giả sử follows đã được cập nhật)
    follows = Follow.query.all()
    data = {'follower_id': [follow.follower_id for follow in follows],
            'following_id': [follow.following_id for follow in follows]}
    df = pd.DataFrame(data)
    user_matrix = pd.crosstab(df['follower_id'], df['following_id']).values

    # Chuẩn hóa ma trận để tránh ảnh hưởng của kích thước
    user_matrix_normalized = normalize(user_matrix, axis=1)

    # Tính toán cosine similarity
    cosine_sim = cosine_similarity(user_matrix_normalized)

    # Chọn người dùng cụ thể
    user_specific_index = df['follower_id'].unique().tolist().index(user_id)

    # Lấy điểm tương đồng giữa người dùng cụ thể và tất cả người dùng khác
    similarities = cosine_sim[user_specific_index]

    # Lọc và đề xuất người dùng mới
    recommended_users_indices = [i for i, sim in enumerate(similarities) if sim > 0 and i != user_specific_index]
    recommended_users_ids = df['follower_id'].unique()[recommended_users_indices]

    # Lọc và đề xuất người dùng mới
    recommended_users = User.query.filter(User.id.in_(recommended_users_ids)).all()

    # Kiểm tra xem người dùng đã theo dõi họ từ danh sách đề xuất chưa
    followed_users_ids = [follow.following_id for follow in Follow.query.filter_by(follower_id=user_id).all()]

    # Loại bỏ những người dùng đã theo dõi từ danh sách đề xuất
    recommended_users = [user for user in recommended_users if user.id not in followed_users_ids]

    return recommended_users
def recommend_hashtags(user_id):
    views = ViewHistory.query.all()
    hashtags_associations = ImageHashtags.query.all()

    # Create DataFrames from the models
    views_df = pd.DataFrame([(view.user_id, view.image_id) for view in views], columns=['user_id', 'image_id'])
    hashtags_associations_df = pd.DataFrame([(assoc.image_id, assoc.hashtag_id) for assoc in hashtags_associations],
                                            columns=['image_id', 'hashtag_id'])

    # Merge DataFrames to create a user - hashtag matrix
    df = pd.merge(views_df, hashtags_associations_df, on='image_id')
    df = df.dropna()

    # Xây dựng ma trận người dùng - hashtag
    user_hashtag_matrix = pd.crosstab(df['user_id'], df['hashtag_id']).values

    # Chuẩn hóa ma trận để tránh ảnh hưởng của kích thước
    user_hashtag_matrix_normalized = normalize(user_hashtag_matrix, axis=1)

    # Tính toán cosine similarity
    cosine_sim = cosine_similarity(user_hashtag_matrix_normalized)


    # Chọn người dùng cụ thể
    user_specific_index = df['user_id'].unique().tolist().index(user_id)

    # Lấy điểm tương đồng giữa người dùng cụ thể và tất cả người dùng khác
    similarities = cosine_sim[user_specific_index]

    # Lọc và đề xuất người dùng tương đồng
    similar_users_indices = [i for i, sim in enumerate(similarities) if sim > 0 and i != user_specific_index]
    similar_users_ids = df['user_id'].unique()[similar_users_indices]

    # Lấy lịch sử hashtag của người dùng tương đồng
    similar_users_hashtags = df[df['user_id'].isin(similar_users_ids)]['hashtag_id'].unique()

    # Loại bỏ những hashtag đã xuất hiện trong lịch sử của người dùng
    recommended_hashtags = [hashtag for hashtag in similar_users_hashtags if
                            hashtag not in df[df['user_id'] == user_id]['hashtag_id'].unique()]
    return recommended_hashtags

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']

        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')

        new_user = User(username=username, password=hashed_password, email=email,type=2)

        try:
            db.session.add(new_user)
            db.session.commit()
            return redirect(url_for('login'))  # Redirect to login page after successful registration
        except Exception as ex:
            return f"Error occurred: {ex}"

    return render_template('register.html')  # Create a register.html file for the registration form
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = User.query.filter_by(username=username).first()

        if user:
            print(f"Entered username: {username}")
            print(f"Hashed password from the database: {user.password}")
            print(f"Hashed password entered during login: {generate_password_hash(password, method='pbkdf2:sha256')}")
            if bcrypt.check_password_hash(user.password, password):
                # Login successful
                session['user_id'] = user.id
                session['username'] = user.username
                session['avatar'] = user.avatar
                if user.type==3:
                    return render_template('accountblock.html')
                else:
                    return redirect(url_for('profile'))
                    return "Login successful"
            else:
                # Login failed
                return "Login failed. Passwords do not match."

            # Handle the case where the username does not exist
        return "Login failed. User not found."

    return render_template('login.html')
@app.route('/update_avatar', methods=['GET', 'POST'])
def update_avatar():
    if 'user_id' in session:
        user_id = session['user_id']
        user = User.query.get(user_id)

        if request.method == 'POST':
            # Check if the post request has the file part
            if 'avatar' not in request.files:
                return redirect(request.url)

            avatar = request.files['avatar']

            # If the user does not select a file, the browser submits an empty file without a filename
            if avatar.filename == '':
                return redirect(request.url)

            # Check if the file is allowed
            if avatar and allowed_file(avatar.filename):
                # Securely save the file with a unique filename
                filename = secure_filename(avatar.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                avatar.save(filepath)

                # Update the user's avatar field with only the filename
                user.avatar = filename
                db.session.commit()

                return redirect(url_for('profile'))

        return render_template('update_avatar.html', user=user)
    else:
        return redirect(url_for('login'))
@app.route('/profile')
def profile():
    if 'user_id' in session:
        user_id = session['user_id']
        user = User.query.get(user_id)
        albums = user.albums if user else []
        saved_images = user.saved_images
        pin_boards = PinBoard.query.filter_by(user_id=user_id).all()
        return render_template('profile2.html', user=user,current_user=user,albums=albums,saved_images = saved_images,pin_boards=pin_boards)
    else:
        return redirect(url_for('login'))
@app.route('/logout')
def logout():
    # Clear the user session
    session.pop('user_id', None)
    session.pop('username', None)

    # Redirect to the login page
    return redirect(url_for('login'))
@app.route('/')
def index():
    # ... (previous code)

    if 'user_id' in session:
        current_user_id = session['user_id']
        current_user = User.query.get(current_user_id)
        images = Image.query.all()
        liked_image_ids = set()

        if current_user:
            liked_image_ids = {like.image_id for like in current_user.likes}

        return render_template('index2.html', images=images, liked_image_ids=liked_image_ids, current_user=current_user)
    else:
        return redirect(url_for('login'))

# ... (previous code)

@app.route('/like_image/<int:image_id>', methods=['POST'])
def like_image(image_id):
    if 'user_id' in session:
        user_id = session['user_id']
        user = User.query.get(user_id)
        image = Image.query.get(image_id)

        if user and image:
            if not user.has_liked(image):
                like = Like(user=user, image=image)
                db.session.add(like)
                db.session.commit()
                flash('Image liked successfully')
            else:
                flash('You have already liked this image')

    return redirect(url_for('view_image', image_id=image_id))
# ... (previous code)

@app.route('/unlike_image/<int:image_id>', methods=['POST'])
def unlike_image(image_id):
    if 'user_id' in session:
        user_id = session['user_id']
        user = User.query.get(user_id)
        image = Image.query.get(image_id)

        if user and image:
            user.unlike(image)
            flash('Image unliked successfully')

    return redirect(url_for('view_image', image_id=image_id))

@app.route('/home')
def home():
    if 'user_id' in session:
        current_user_id = session['user_id']
        current_user = User.query.get(current_user_id)
        users = User.query.filter(User.type == 2,User.id != current_user_id).all()

        # Get a list of user IDs that the current user is following
        following_ids = [follow.following_id for follow in current_user.following]

        return render_template('homes.html', users=users, current_user=current_user, following_ids=following_ids)
    else:
        return redirect(url_for('login'))


@app.route('/follow/<int:user_id>', methods=['POST'])
def follow(user_id):
    if 'user_id' in session:
        current_user_id = session['user_id']
        current_user = User.query.get(current_user_id)
        user_to_follow = User.query.get(user_id)

        if current_user and user_to_follow:
            current_user.follow(user_to_follow)
            flash('You are now following {}'.format(user_to_follow.username))
            return redirect(url_for('view_profile', user_id=user_to_follow.id))

    return redirect(url_for('login'))

@app.route('/unfollow/<int:user_id>', methods=['POST'])
def unfollow(user_id):
    if 'user_id' in session:
        current_user_id = session['user_id']
        current_user = User.query.get(current_user_id)
        user_to_unfollow = User.query.get(user_id)

        if current_user and user_to_unfollow:
            current_user.unfollow(user_to_unfollow)
            flash('You have unfollowed {}'.format(user_to_unfollow.username))
            return redirect(url_for('view_profile', user_id=user_to_unfollow.id))

    return redirect(url_for('login'))


# ... (previous code)

@app.route('/following')
def following():
    if 'user_id' in session:
        current_user_id = session['user_id']
        current_user = User.query.get(current_user_id)

        if current_user:
            following_users = current_user.following.all()
            return render_template('following.html', following_users=following_users)

    return redirect(url_for('login'))
@app.route('/follower')
def follower():
    if 'user_id' in session:
        current_user_id = session['user_id']
        current_user = User.query.get(current_user_id)

        if current_user:
            following_users = current_user.followers.all()
            return render_template('following.html', following_users=following_users)

    return redirect(url_for('login'))
from flask import request, flash, redirect, render_template, url_for

@app.route('/upload_image', methods=['GET', 'POST'])
def upload_image():
    if 'user_id' in session:
        user_id = session['user_id']
        user = User.query.get(user_id)

        if request.method == 'POST':
            if 'image' in request.files:
                image = request.files['image']

                if image.filename == '':
                    flash('No selected file')
                    return redirect(request.url)

                if image and allowed_file(image.filename):
                    filename = secure_filename(image.filename)
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

                    img_path = Path(filepath)

                    image.save(filepath)
                    feature = fe.extract(img=PILImage.open(img_path))
                    feature_path = destination_path / (Path(filepath).stem + ".npy")
                    np.save(feature_path, feature)
                    caption = request.form.get('caption', '')
                    description = request.form.get('description', '')
                    link = request.form.get('link', '')
                    allow_comments = request.form.get('allow_comments')
                    allow_comments = allow_comments == 'on'
                    album_id = request.form.get('album_id')
                    album_id = int(album_id) if album_id else None
                    new_image = Image(user=user, image_path=filename, caption=caption,description=description,link=link, album_id=album_id,allow_comments=allow_comments)
                    db.session.add(new_image)
                    db.session.commit()
                    hashtags_input = request.form.getlist('hashtags') or ''
                    if hashtags_input:
                        hashtags = [tag.strip() for tag in hashtags_input]

                        # Xử lý hashtags như cần thiết (ví dụ: tạo hashtag mới nếu chưa tồn tại)
                        for tag in hashtags:
                            hashtag = Hashtag.query.filter_by(name=tag).first()
                            if not hashtag:
                                hashtag = Hashtag(name=tag)
                                db.session.add(hashtag)
                                db.session.commit()
                                print(f"Added new hashtag: {tag}")

                            # Liên kết hashtag với ảnh
                            image_hashtags = ImageHashtags(image_id=new_image.id, hashtag_id=hashtag.id)
                            db.session.add(image_hashtags)
                            db.session.commit()
                            print(f"Associated hashtag {tag} with the image")

                    flash('Image uploaded successfully')
                    return redirect(url_for('profile'))

        # If it's not a POST request or after handling the POST request, get user albums
        albums = user.albums if user else []
        all_hashtags = Hashtag.query.all()
        return render_template('upload_image.html', user=user, albums=albums, all_hashtags=all_hashtags)

    else:
        return redirect(url_for('login'))

@app.route('/get_hashtag_suggestions')
def get_hashtag_suggestions():
    input_text = request.args.get('input', '')

    # Perform a query to get hashtag suggestions based on the input_text
    suggestions = Hashtag.query.filter(Hashtag.name.ilike(f'%{input_text}%')).all()

    # Return the suggestions as JSON
    return jsonify([{'name': hashtag.name} for hashtag in suggestions])

@app.route('/profile/update_caption/<int:image_id>', methods=['GET', 'POST'])
def update_caption(image_id):
    if 'user_id' in session:
        current_user_id = session['user_id']
        current_user = User.query.get(current_user_id)

        if current_user:
            image = Image.query.get(image_id)

            if request.method == 'POST':
                new_caption = request.form['new_caption']
                new_description = request.form['new_description']
                new_link = request.form['new_link']
                image.caption = new_caption
                image.description=new_description
                image.link=new_link
                allow_comments = request.form.get('allow_comments')
                image.allow_comments = allow_comments == 'on'
                album_id = request.form.get('album_id')
                album_id = int(album_id) if album_id else None
                image.album_id = album_id
                db.session.commit()

                hashtags_input = request.form.getlist('hashtags') or ''
                for tag in hashtags_input:
                    if tag:  # Đảm bảo rằng hashtag không rỗng
                        hashtag = Hashtag.query.filter_by(name=tag).first()
                        if not hashtag:
                            hashtag = Hashtag(name=tag)
                            db.session.add(hashtag)
                            db.session.commit()
                            print(f"Added new hashtag: {tag}")

                        # Liên kết hashtag với ảnh
                        image_hashtags = ImageHashtags(image_id=image.id, hashtag_id=hashtag.id)
                        db.session.add(image_hashtags)
                        db.session.commit()
                        print(f"Associated hashtag {tag} with the image")

                # Gắn danh sách các hashtag vào ảnh

                flash('Caption updated successfully')
                return redirect(url_for('profile'))
            albums = current_user.albums if current_user else []
            existing_hashtags = [h.name for h in image.hashtags]
            all_hashtags = Hashtag.query.all()
            return render_template('update_caption.html', image=image,albums=albums, existing_hashtags=existing_hashtags,all_hashtags =all_hashtags )

    return redirect(url_for('login'))


@app.route('/profile/delete_image/<int:image_id>', methods=['POST'])
def delete_image(image_id):
    if 'user_id' in session:
        current_user_id = session['user_id']
        current_user = User.query.get(current_user_id)

        if current_user:
            image = Image.query.get(image_id)
            db.session.delete(image)
            db.session.commit()
            flash('Image deleted successfully')
            return redirect(url_for('profile'))

    return redirect(url_for('login'))
# ... (previous code)

@app.route('/view_profile/<int:user_id>')
def view_profile(user_id):
    viewed_user = User.query.get(user_id)

    if 'user_id' in session:
        current_user_id = session['user_id']
        current_user = User.query.get(current_user_id)

        if viewed_user:
            return render_template('view_profile.html', viewed_user=viewed_user, current_user=current_user)
        else:
            flash('User not found')
            return redirect(url_for('index'))

    return redirect(url_for('login'))
@app.route('/view_image_comment/<int:image_id>', methods=['GET', 'POST'])
def view_image_comment(image_id):
    image = Image.query.get(image_id)

    if image:
        if request.method == 'POST':
            if 'user_id' in session:
                user_id = session['user_id']
                user = User.query.get(user_id)
                text = request.form['comment_text']

                if text:
                    comment = Comment(user=user, image=image, comment_text=text)
                    db.session.add(comment)
                    db.session.commit()
                    flash('Comment added successfully')

        return render_template('view_image.html', image=image, current_user=user)
    else:
        flash('Image not found')
        return redirect(url_for('index'))
@app.route('/view_image/<int:image_id>', methods=['GET'])
def view_image(image_id):
    if 'user_id' in session:
        user_id = session['user_id']
        user = User.query.get(user_id)
        image = Image.query.get(image_id)
        pin_boards = PinBoard.query.all()
        print(pin_boards)
        if user and image:
            # Thêm dòng vào bảng lịch sử
            view_history = ViewHistory(user=user, image=image)
            db.session.add(view_history)
            image.views += 1
            db.session.commit()
            num_likes = Like.query.filter_by(image_id=image.id).count()

            return render_template('view_image.html', current_user=user, image=image,num_likes=num_likes,pin_boards=pin_boards)
        else:
            flash('Image or user not found.')
    else:
        flash('Please log in to view images.')

    return redirect(url_for('login'))

@app.route('/generate_qr_code/<int:image_id>', methods=['GET'])
def generate_qr_code(image_id):
    image = Image.query.get(image_id)

    if image:
        # Generate a QR code for the detailed link
        detailed_link = url_for('view_image', image_id=image_id, _external=True)
        print(detailed_link)
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(detailed_link)
        qr.make(fit=True)

        qr_img = qr.make_image(fill_color="black", back_color="white")

        # Save the QR code image to a file
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], f'qrcode_{image_id}.png')
        qr_img.save(save_path)

        # Return the QR code image as a response
        return send_file(save_path, mimetype='image/png')
    else:
        flash('Image not found.')

    return redirect(url_for('index'))
@app.route('/view_history')
def view_history():
    if 'user_id' in session:
        user_id = session['user_id']
        user = User.query.get(user_id)

        if user:
            # Truy vấn lịch sử xem ảnh của người dùng và lấy bức ảnh mới nhất
            latest_view = db.session.query(ViewHistory, func.max(ViewHistory.timestamp).label('max_timestamp')) \
                .filter_by(user_id=user.id) \
                .group_by(ViewHistory.image_id) \
                .order_by(func.max(ViewHistory.timestamp).desc()) \
                .first()

            if latest_view:
                latest_image_id = latest_view[0].image_id
                latest_image = Image.query.get(latest_image_id)

                return render_template('view_history.html', user=user, latest_image=latest_image)
            else:
                # Không có lịch sử xem
                return render_template('view_history.html', user=user, latest_image=None)

    return redirect(url_for('login'))
@app.route('/create_album', methods=['GET', 'POST'])
def create_album():
    if request.method == 'POST':
        user_id = session.get('user_id')  # Get the user ID from the session
        user = User.query.get(user_id)

        if user:
            album_name = request.form['album_name']
            new_album = Album(user=user, name=album_name)
            db.session.add(new_album)
            db.session.commit()
            return redirect(url_for('profile'))  # Redirect to the user's profile or wherever you want
        else:
            return "User not found"

    return render_template('create_album.html')
@app.route('/albums')
def display_albums(album_id):
    user_id = session.get('user_id')
    if user_id:
        user = User.query.get(user_id)
        albums = user.albums  # Directly access the relationship attribute
        album = Album.query.get(album_id)
        return render_template('albums.html', user=user, albums=albums,album=album,album_id=album_id)
    else:
        return redirect(url_for('login'))


@app.route('/update_album/<int:album_id>', methods=['GET', 'POST'])
def update_album(album_id):
    user_id = session.get('user_id')
    user = User.query.get(user_id)
    album = Album.query.get(album_id)

    if request.method == 'POST':
        if user and album and album.user == user:
            album_name = request.form['album_name']
            album.name = album_name
            db.session.commit()
            return redirect(url_for('profile'))
        else:
            return "Album not found or unauthorized"

    return render_template('update_album.html', album=album)
@app.route('/delete_album/<int:album_id>', methods=['POST'])
def delete_album(album_id):
    user_id = session.get('user_id')
    user = User.query.get(user_id)
    album = Album.query.get(album_id)

    if user and album and album.user == user:
        db.session.delete(album)
        db.session.commit()
        return redirect(url_for('profile'))
    else:
        return "Album not found or unauthorized"

# Assuming you have the necessary imports
from flask import render_template

@app.route('/album/<int:album_id>')
def view_album(album_id):
    if 'user_id' in session:
        user_id = session['user_id']
        user = User.query.get(user_id)

        if user:
            album = Album.query.get(album_id)

            if album:
                images = Image.query.filter_by(album_id=album.id).all()
                return render_template('view_albums.html', user=user, album=album, images=images)

    return redirect(url_for('login'))
@app.route('/delete_image_from_album/<int:image_id>/<int:album_id>', methods=['POST'])
def delete_image_from_album(image_id, album_id):
    if 'user_id' in session:
        user_id = session['user_id']
        user = User.query.get(user_id)

        if user:
            image = Image.query.get(image_id)

            if image and image.album_id == album_id:
                # Delete the image from the album
                image.album_id = None
                db.session.commit()

    return redirect(url_for('view_album', album_id=album_id))

@app.route('/searchimage', methods=['GET', 'POST'])
def searchimage():
    if request.method == 'POST':
        file = request.files['query_img']
        # Save query image
        img = PILImage.open(file.stream)  # PIL image
        uploaded_img_path = "static/upim/" + datetime.now().isoformat().replace(":", ".") + "_" + file.filename
        img.save(uploaded_img_path)
        # Run search
        query = fe.extract(img)
        dists = np.linalg.norm(features-query, axis=1)  # L2 distances to features
        ids = np.argsort(dists)[:5]  # Top 30 results
        scores = [(Path(img_paths[id]).name) for id in ids]
        print(scores)

        images = Image.query.filter(Image.image_path.in_(scores)).order_by(Image.timestamp.desc()).all()
        print(images)
        # image_paths = [image.image_path for image in images]
        result_data = db.session.query(Image.id, Image.link,Image.image_path, User.username, User.id).join(User).filter(
            Image.image_path.in_(scores)).all()
        image_data = [{"id": id,"link":link,"image_path": image_path,"username": username, "user_id": user_id} for
                       id ,link ,image_path,username, user_id  in result_data]
        return render_template('searchimage.html',
                               query_path=uploaded_img_path,
                               scores=scores, image_data=image_data)
    else:
        return render_template('searchimage.html')


@app.route('/following_images')
def following_images():
    if 'user_id' in session:
        if 'user_id' in session:
            current_user_id = session['user_id']
            current_user = User.query.get(current_user_id)

            if current_user:
                # Get the IDs of users being followed by the current user
                following_ids = [follow.following_id for follow in current_user.following]

                # Retrieve the users being followed
                following_users = User.query.filter(User.id.in_(following_ids)).all()

                # Retrieve images from the users the current user is following
                following_images = []
                for followed_user in following_users:
                    # Extend the list with the images of each followed user
                    following_images.extend(followed_user.images)

            return render_template('following_images.html', following_users=following_users, following_images=following_images)

    return redirect(url_for('login'))


@app.route('/search_history', methods=['GET', 'POST'])
def search_history():
    if 'user_id' in session:
        user_id = session['user_id']
        user = User.query.get(user_id)
        users = User.query.filter(User.type == 2, User.id != user_id).all()

        # Get a list of user IDs that the current user is following
        following_ids = [follow.following_id for follow in user.following]
        recommend_users = recommend_userss(user_id)
        suggested_pin_boards = user.suggested_pin_boards()
        suggested_hashtags_collaborative = recommend_hashtags(user_id)
        print("Recommended Hashtags:", suggested_hashtags_collaborative)
        recommended_hashtags_objects = Hashtag.query.filter(Hashtag.id.in_(suggested_hashtags_collaborative)).all()
        # Chuyển kết quả đề xuất vào template và hiển thị
        if user:
            # Truy vấn lịch sử xem ảnh của người dùng và lấy bức ảnh mới nhất
            latest_view = db.session.query(ViewHistory, func.max(ViewHistory.timestamp).label('max_timestamp')) \
                .filter_by(user_id=user.id) \
                .group_by(ViewHistory.image_id) \
                .order_by(func.max(ViewHistory.timestamp).desc()) \
                .first()

            if latest_view:
                latest_image_id = latest_view[0].image_id
                latest_image = Image.query.get(latest_image_id)
                query_image_path = os.path.join('static', 'uploads', latest_image.image_path).replace("\\", "/")

                query = fe.extract(PILImage.open(query_image_path)) # PIL image
                dists = np.linalg.norm(features - query, axis=1)  # L2 distances to features
                ids = np.argsort(dists)[:7]  # Top 5 results
                scores = [(Path(img_paths[id]).name) for id in ids]
                result_data = db.session.query(Image.image_path, User.username, User.id, Image.id) \
                    .join(User) \
                    .filter(Image.image_path.in_(scores)) \
                    .all()

                image_data = [{"image_path": image_path, "username": username, "user_id": user_id, "id": image_id} for
                              image_path, username, user_id,image_id in result_data]

                return render_template('search_history.html', current_user=user, latest_image=latest_image,query_path=latest_image.image_path,
                                           scores=scores, image_data=image_data,users=users ,following_ids=following_ids,recommend_users=recommend_users,recommended_hashtags_objects=recommended_hashtags_objects,suggested_pin_boards=suggested_pin_boards)
            else:
                # Không có lịch sử xem
                return render_template('search_history.html', user=user, latest_image=None)

    return redirect(url_for('login'))



@app.route('/save_image/<int:image_id>', methods=['POST'])
def save_image(image_id):
    if 'user_id' in session:
        user_id = session['user_id']
        user = User.query.get(user_id)
        image = Image.query.get(image_id)

        if user and image:
            user.save_image(image)
            flash('Image saved successfully')

    return redirect(url_for('view_image',image_id=image_id))

@app.route('/unsave_image/<int:image_id>', methods=['POST'])
def unsave_image(image_id):
    if 'user_id' in session:
        user_id = session['user_id']
        user = User.query.get(user_id)
        image = Image.query.get(image_id)

        if user and image:
            user.unsave_image(image)
            flash('Image unsaved successfully')

    return redirect(url_for('view_image',image_id=image_id))
@app.route('/saved_images')
def saved_images():
    if 'user_id' in session:
        user_id = session['user_id']
        user = User.query.get(user_id)

        if user:
            saved_images = user.saved_images
            return render_template('save_images.html', user=user, saved_images=saved_images)

    return redirect(url_for('login'))
# In your Flask app




@app.route('/report_image/<int:image_id>', methods=['GET', 'POST'])
def report_image(image_id):
    if 'user_id' in session:
        user_id = session['user_id']
        user = User.query.get(user_id)
        if request.method == 'POST':
            reason = request.form.get('reason')

            if reason:
                report = ReportImage(user_id=user.id, image_id=image_id, reason=reason)
                db.session.add(report)
                db.session.commit()

                flash('Image reported successfully.')
                return redirect(url_for('index'))

    return render_template('report_image.html', image_id=image_id)
@app.route('/reported_images', methods=['GET'])
def reported_images():
    # Query reported images from the database
    reported_images = ReportImage.query.all()

    # Render the template with reported images
    return render_template('reported_images.html', reported_images=reported_images)

@app.route('/edit_img')
def editimg():
    return render_template('edit_img.html')
@app.route("/transfer", methods=['GET', 'POST'])
def transfer():
    if request.method == 'POST':
        target = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static/images/')
        if not os.path.isdir(target):
            os.mkdir(target)

        data = request.form.get("style")
        my_files = []

        for file in request.files.getlist("file"):
            filename = file.filename
            destination = os.path.join(target, filename)
            file.save(destination)
            my_files.append(filename)

        # Assuming neuralStyleTransfer returns the processed image filename
        new_img = neuralStyleProcess.neuralStyleTransfer(target, my_files[0], data)

        return render_template("completetransfer.html", image_names=my_files, selected_style=data, processed_image=new_img)

    return render_template("style_transfer.html")


@app.route('/blackwhite', methods=['GET', 'POST'])
def blackwhite():
    if request.method == 'GET':
        return render_template("blackwhite.html", check=0)
    else:
        if 'InputImg' not in request.files:
            print("No file part")
            return redirect(request.url)
        file = request.files['InputImg']
        if file.filename == '':
            print('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filestr = request.files['InputImg'].read()
            npimg = np.fromstring(filestr, np.uint8)
            image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            image = cv2.resize(image, (128, 128))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

            scaled = image.astype("float32") / 255.0
            lab = cv2.cvtColor(scaled, cv2.COLOR_RGB2LAB)
            resized = cv2.resize(lab, (224, 224))
            L = cv2.split(resized)[0]
            L -= 50

            net.setInput(cv2.dnn.blobFromImage(L))
            ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
            ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

            L = cv2.split(lab)[0]
            colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

            colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2RGB)
            colorized = np.clip(colorized, 0, 1)
            colorized = (255 * colorized).astype("uint8")

            pil_img = PILImage.fromarray(colorized)
            buff = io.BytesIO()
            pil_img.save(buff, format="JPEG")
            new_image_string = base64.b64encode(buff.getvalue()).decode("utf-8")
            pil_original_img = PILImage.fromarray(image)
            buff_original = io.BytesIO()
            pil_original_img.save(buff_original, format="JPEG")
            original_image_string = base64.b64encode(buff_original.getvalue()).decode("utf-8")
        return render_template('blackwhite.html', img=new_image_string, oimg=original_image_string, check=1)
@app.route('/imagecap', methods=['GET','POST'])
def imagecap():
    return render_template('imagecap.html')
@app.route('/imagecaps', methods=['GET','POST'])
def imagecaps():
    try:
        file = request.files['file']
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        result = classify(file_path)  # Your classification function
        print(result)
        return jsonify({'result': result,'file_path': file_path})
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/search_hashtag', methods=['GET', 'POST'])
def search_hashtag():
    unique_hashtags = set(hashtag.name for hashtag in Hashtag.query.all())
    if request.method == 'POST':
        hashtag_name = request.form.getlist('hashtag')


        if hashtag_name:
            # Query images associated with the given hashtag
            images = db.session.query(Image).join(ImageHashtags).join(Hashtag).filter(
                Hashtag.name.in_(hashtag_name)).all()

            return render_template('search_hashtag_result.html', images=images, hashtag=hashtag_name)
    all_hashtags = Hashtag.query.all()
    popular_hashtags = db.session.query(Hashtag.name, func.count(Hashtag.name).label('count'))\
        .join(ImageHashtags).group_by(Hashtag.name)\
        .order_by(func.count(Hashtag.name).desc())\
        .limit(10).all()
    return render_template('search_hashtag.html', hashtags=popular_hashtags, all_hashtags=all_hashtags)
@app.route('/all_hashtags')
def all_hashtags():
    if 'user_id' in session:
        user_id = session['user_id']
        user = User.query.get(user_id)

        if user:
    # Assuming you have a Hashtag class with a 'name' attribute
         unique_hashtags = set(hashtag.name for hashtag in Hashtag.query.all())

    return render_template('all_hashtags.html', hashtags=unique_hashtags)
@app.route('/hashtag/<string:hashtag_name>')
def images_by_hashtag(hashtag_name):
    hashtag = Hashtag.query.filter_by(name=hashtag_name).first()
    if hashtag:
        images = hashtag.images
        return render_template('images_by_hashtag.html', hashtag=hashtag, images=images)
    else:
        flash('Hashtag not found', 'error')
        return redirect(url_for('all_hashtags'))
@app.route('/filter')
def filter():
    return render_template('filter.html')
@app.route('/accountblock')
def accountblock():
    return render_template('accountblock.html')
@app.route('/filters', methods=['GET','POST'])
def filters():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file:
        # Read the uploaded image
        img = PILImage.open(file)
        img_array = np.array(img)

        # Get the selected effect from the form
        selected_effect = request.form['effect']
        input_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'input.png')
        img.save(input_filename)

        # Apply the selected effect
        if selected_effect == 'cartoon':
            # Convert to color (if not already in color)
            img_array = cartoonize_image(img_array)
        elif selected_effect == 'pixel':
            img_array = pixelize_image(img_array, pixel_size=10)
        elif selected_effect=='daylight':
            img_array= adjust_lightness(img_array, 1.5)
        elif selected_effect=='emboss':
            img_array= apply_emboss_effect(img_array)
        elif selected_effect=='tv60':
            img_array= tv60(img_array)
        elif selected_effect=='pencil_sketch_color':
            img_array= pencil_sketch_col(img_array)
        elif selected_effect=='pencil_sketch_grey':
            img_array= pencil_sketch_grey(img_array)
        elif selected_effect == 'sharpen':
            img_array = sharpen(img_array)
        elif selected_effect == 'HDR':
            img_array = HDR(img_array)
        elif selected_effect == 'invert':
            img_array = invert(img_array)
        elif selected_effect == 'Winter':
            img_array = Winter(img_array)
        elif selected_effect == 'Summer':
            img_array = Summer(img_array)
        elif selected_effect == 'sepia':
            img_array = sepia(img_array)

        # Create a new Image object from the manipulated pixel array
        result_img = PILImage.fromarray(img_array.astype('uint8'))

        # Save the result image to the server's file system
        result_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'result.png')
        result_img.save(result_filename)


        return render_template('filter.html', input_filename=input_filename,result_filename=result_filename)

@app.route('/qr_scanner')
def qr_scanner():
    return render_template('qr_scanner.html')
def generate_confirmation_code():
    return str(random.randint(100000, 999999))
def send_password_reset_email(email, confirmation_code):
    msg = Message('Password Reset Confirmation Code', sender=app.config['MAIL_DEFAULT_SENDER'], recipients=[email])
    msg.body = f'Your confirmation code is: {confirmation_code}'
    mail.send(msg)
@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form.get('email')
        user = User.query.filter_by(email=email).first()

        if user:
            # Generate and save a confirmation code for the user
            confirmation_code = generate_confirmation_code()
            user.password_reset_code = confirmation_code
            db.session.commit()

            # Send the confirmation code to the user's email
            send_password_reset_email(user.email, confirmation_code)

            flash('An email with instructions to reset your password has been sent.')
            return redirect(url_for('reset_password'))
        else:
            flash('Email address not found.')
            return render_template('forgot_password.html')

    return render_template('forgot_password.html')


# Define a route to handle the password reset confirmation
@app.route('/reset_password', methods=['GET', 'POST'])
def reset_password():
    if request.method == 'POST':
        confirmation_code = request.form.get('confirmation_code')
        user = User.query.filter_by(password_reset_code=confirmation_code).first()

        if user:
            # Reset the password
            new_password = request.form.get('new_password')
            hashed_password = bcrypt.generate_password_hash(new_password).decode('utf-8')
            user.password = hashed_password
            user.password_reset_code = None
            db.session.commit()

            flash('Your password has been successfully reset. You can now log in with your new password.')
            return redirect(url_for('login'))
        else:
            flash('Invalid confirmation code.')
            return render_template('reset_password.html')

    return render_template('reset_password.html')

# ...

# Define a route to handle changing the password
# ...

# Define a route to handle changing the password
@app.route('/change_password', methods=['GET', 'POST'])
def change_password():
    user_id = session.get('user_id')  # Replace with your actual session key for user identification

    if not user_id:
        flash('You must be logged in to change your password.')
        return redirect(url_for('login'))

    user = User.query.get(user_id)

    if not user:
        flash('Invalid user. Please log in again.')
        return redirect(url_for('login'))

    if request.method == 'POST':
        current_password = request.form.get('current_password')
        new_password = request.form.get('new_password')
        confirm_new_password = request.form.get('confirm_new_password')


        if not user or not bcrypt.check_password_hash(user.password, current_password):
            flash('Incorrect current password. Please try again.')
            return render_template('change_password.html')

        if new_password != confirm_new_password:
            flash('New password and confirmation do not match. Please try again.')
            return render_template('change_password.html')

        # Update the password
        hashed_password = bcrypt.generate_password_hash(new_password).decode('utf-8')
        user.password = hashed_password
        db.session.commit()

        flash('Your password has been successfully changed.')
        return redirect(url_for('profile'))

    return render_template('change_password.html')

@app.route('/suggested_users')
def suggested_users():
    if 'user_id' in session:
        user_id = session['user_id']
        current_user = User.query.get(user_id)

        # Get suggested users
        suggested_users = current_user.suggested_users()

        return render_template('suggested_users.html', suggested_users=suggested_users)

    # Handle the case when the user is not logged in
    flash('Please log in to see suggested users.')
    return redirect(url_for('login'))
import pandas as pd



@app.route('/recommendations')
def recommendations():
    if 'user_id' in session:
        user_id = session['user_id']
        user = User.query.get(user_id)
        users = User.query.filter(User.type == 2, User.id != user_id).all()

        # Get a list of user IDs that the current user is following
        following_ids = [follow.following_id for follow in user.following]
        recommend_users=recommend_userss(user_id)
        return render_template('recommendations.html', user=users, recommended_items=recommend_users,following_ids=following_ids)

    else:
        return redirect(url_for('login'))




@app.route('/suggest_hashtags', methods=['GET'])
def suggest_hashtags():
    if 'user_id' in session:
        user_id = session['user_id']
        user = User.query.get(user_id)

    # Gọi hàm để đề xuất các hashtag sử dụng Collaborative Filtering
        suggested_hashtags_collaborative = recommend_hashtags(user_id)
        print("Recommended Hashtags:", suggested_hashtags_collaborative )
        recommended_hashtags_objects = Hashtag.query.filter(Hashtag.id.in_(suggested_hashtags_collaborative)).all()
    # Chuyển kết quả đề xuất vào template và hiển thị
    return render_template('suggest_hashtags.html', suggested_hashtags=recommended_hashtags_objects)

@app.route('/add_pin_board', methods=['GET', 'POST'])
def add_pin_board():
    if 'user_id' in session:
        user_id = session['user_id']
        if request.method == 'POST':
            # Lấy dữ liệu từ form
            # Thay bằng cách lấy ID của người dùng từ đâu đó (session, login, ...)
            name = request.form['pin_board_name']

            # Tạo một bảng ghim mới
            new_pin_board = PinBoard(user_id=user_id, name=name)

            # Thêm vào cơ sở dữ liệu
            db.session.add(new_pin_board)
            db.session.commit()

            # Chuyển hướng về trang hiển thị danh sách bảng ghim
            return redirect(url_for('list_pin_boards'))

    return render_template('add_pin_board.html')

# Route để hiển thị danh sách các bảng ghim
@app.route('/list_pin_boards')
def list_pin_boards():
    # Lấy danh sách các bảng ghim từ cơ sở dữ liệu
    pin_boards = PinBoard.query.all()

    return render_template('list_pin_boards.html', pin_boards=pin_boards)

@app.route('/add_image_to_pin_board/<int:image_id>', methods=['POST'])
def add_image_to_pin_board(image_id):
    if request.method == 'POST':
        # Lấy dữ liệu từ form
        pin_board_id = request.form['pin_board_id']

        existing_pinned_image = PinnedImage.query.filter_by(image_id=image_id, pin_board_id=pin_board_id).first()
        if existing_pinned_image:
            flash('Image is already in the Pin Board', 'warning')
        else:
            # Tạo một PinnedImage mới
            new_pinned_image = PinnedImage(image_id=image_id, pin_board_id=pin_board_id)

            # Thêm vào cơ sở dữ liệu
            db.session.add(new_pinned_image)
            db.session.commit()

            flash('Image added to Pin Board successfully', 'success')
        return redirect(url_for('view_image', image_id=image_id))

@app.route('/view_pin_board/<int:pin_board_id>', methods=['GET'])
def view_pin_board(pin_board_id):
    if 'user_id' in session:
        user_id = session['user_id']
        user = User.query.get(user_id)
        pin_board = PinBoard.query.get(pin_board_id)

        if user and pin_board:
            # Lấy danh sách các ảnh được ghim trong bảng ghim cụ thể
            pinned_images = PinnedImage.query.filter_by(pin_board_id=pin_board.id).all()

            # Lấy danh sách các đối tượng Image tương ứng với các ảnh được ghim
            images = [Image.query.get(pinned_image.image_id) for pinned_image in pinned_images]

            return render_template('view_pin_board.html', current_user=user, pin_board=pin_board, pinned_images=pinned_images, images=images)
        else:
            flash('Pin board or user not found.')
    else:
        flash('Please log in to view pin boards.')

    return redirect(url_for('login'))
@app.route('/view_pin_board_user/<int:pin_board_id>', methods=['GET'])
def view_pin_board_user(pin_board_id):
    if 'user_id' in session:
        user_id = session['user_id']
        user = User.query.get(user_id)
        pin_board = PinBoard.query.get(pin_board_id)

        if user and pin_board:
            # Lấy danh sách các ảnh được ghim trong bảng ghim cụ thể
            pinned_images = PinnedImage.query.filter_by(pin_board_id=pin_board.id).all()

            # Lấy danh sách các đối tượng Image tương ứng với các ảnh được ghim
            images = [Image.query.get(pinned_image.image_id) for pinned_image in pinned_images]

            return render_template('view_pinboard_user.html', current_user=user, pin_board=pin_board, pinned_images=pinned_images, images=images)
        else:
            flash('Pin board or user not found.')
    else:
        flash('Please log in to view pin boards.')

    return redirect(url_for('login'))
@app.route('/remove_from_pin_board/<int:pin_board_id>/<int:image_id>', methods=['GET'])
def remove_from_pin_board(pin_board_id, image_id):
    if 'user_id' in session:
        user_id = session['user_id']
        user = User.query.get(user_id)
        pin_board = PinBoard.query.get(pin_board_id)
        image = Image.query.get(image_id)

        if user and pin_board and image:
            # Kiểm tra xem ảnh có trong bảng ghim không
            pinned_image = PinnedImage.query.filter_by(pin_board_id=pin_board.id, image_id=image.id).first()

            if pinned_image:
                # Xóa ảnh khỏi bảng ghim
                db.session.delete(pinned_image)
                db.session.commit()

                flash('Image removed from pin board successfully.')
            else:
                flash('Image is not pinned to the pin board.')

            return redirect(url_for('view_pin_board', pin_board_id=pin_board_id))
        else:
            flash('Pin board, image, or user not found.')
    else:
        flash('Please log in to perform this action.')

    return redirect(url_for('login'))

@app.route('/delete_pin_board/<int:pin_board_id>', methods=['POST'])
def delete_pin_board(pin_board_id):
    if request.method == 'POST':
        # Lấy Pin Board cần xóa
        pin_board = PinBoard.query.get(pin_board_id)

        if pin_board:
            # Xóa tất cả Pinned Images của Pin Board
            PinnedImage.query.filter_by(pin_board_id=pin_board.id).delete()

            # Xóa Pin Board
            db.session.delete(pin_board)
            db.session.commit()

            flash('Pin Board and associated images deleted successfully', 'success')
        else:
            flash('Pin Board not found.', 'warning')

        return redirect(url_for('list_pin_boards'))
@app.route('/search_pin_boards', methods=['GET', 'POST'])
def search_pin_boards():
    if request.method == 'POST':
        keyword = request.form['keyword']

        # Tìm kiếm Pin Boards theo từ khóa
        matching_pin_boards = PinBoard.query.filter(PinBoard.name.ilike(f'%{keyword}%')).all()

        # Tạo một danh sách chứa tất cả các ảnh của các Pin Board
        all_images = []
        for pin_board in matching_pin_boards:
            # Lấy tất cả các PinnedImage của Pin Board
            pinned_images = PinnedImage.query.filter_by(pin_board_id=pin_board.id).all()
            # Lấy tất cả các Image của PinnedImage
            images = [Image.query.get(pinned_image.image_id) for pinned_image in pinned_images]
            all_images.extend(images)

        return render_template('search_pin_boards.html', keyword=keyword, pin_boards=matching_pin_boards, all_images=all_images)

    return render_template('search_pin_boards.html', keyword=None, pin_boards=None, all_images=None)
model_keras = load_model("static/models/generator_model_b.h5")
SIZE =128
def image_upload(path):
    images = []
    images.append(path)
    return np.array(images)

def convert_image_inputs(images):
    labels = []
    for image_path in images:
        img = cv2.imread(image_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # res_img = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_CUBIC)
            res_img = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
            labels.append(res_img)
    return labels, img.shape[1], img.shape[0]


def make_color(path_input):
    image = image_upload(path_input)
    input_img, w, h = convert_image_inputs(image)
    input_img = np.array(input_img)
    input_img = input_img/255.0
    generated_image = model_keras.predict(input_img)
    return generated_image, w, h

def result(output):
    num_samples = len(output)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(output)
    ax.set_title('Output')
    plt.show()


@app.route('/colorimage', methods=['GET', 'POST'])
def colorimage():
    return render_template('colorimage.html')


@app.route('/upcolorimage', methods=['GET', 'POST'])
def upcolorimage():
    if request.method == 'POST':
        if 'image' in request.files:
            image = request.files['image']
            if image.filename != '':
                image_path = os.path.join('static', 'uploads', image.filename)
                image.save(image_path)

                generated_image, img_width, img_height = make_color(image_path)
                print(generated_image[0], img_width, img_height)
                generated_image = generated_image[0]
                generated_image = generated_image * 255.0

                generated_image_filename = 'generated_' + image.filename
                generated_image_path = os.path.join('static', 'generated', generated_image_filename)
                generated_image_rgb = cv2.cvtColor(generated_image, cv2.COLOR_BGR2RGB)
                # SIZE = 1080
                res_img1 = cv2.resize(generated_image_rgb, (img_height, img_width), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(generated_image_path, res_img1)
                session['before_image_url'] = url_for('static', filename='uploads/' + image.filename)
                session['after_image_url'] = url_for('static', filename='generated/' + generated_image_filename)
                return redirect(url_for('show_color_image', filename=generated_image_filename))

    return render_template('resultcolor.html')


@app.route('/show_color_image/<filename>')
def show_color_image(filename):
    uploaded_image_url = url_for('static', filename='generated/' + filename)
    before_image_url = session.get('before_image_url', '')
    after_image_url = session.get('after_image_url', '')
    return render_template('resultcolor.html', uploaded_image_url=uploaded_image_url, before_image_url=before_image_url,
                           after_image_url=after_image_url)

@app.route('/chat/<int:partner_id>', methods=['GET', 'POST'])
def chat(partner_id):
    if 'user_id' in session:
        user_id = session['user_id']
        current_user = User.query.get(user_id)
        partner = User.query.get(partner_id)
        if not partner:
            # Xử lý khi không tìm thấy người dùng đối tác
            return redirect(url_for('homes'))

        if request.method == 'POST':
            content = request.form.get('content')
            current_user.send_message(partner, content)
            # Cập nhật lại trang để hiển thị tin nhắn mới
            return redirect(url_for('chat', partner_id=partner.id))

        messages = current_user.get_messages(partner_id)
        return render_template('chat.html', partner=partner, messages=messages,current_user=current_user)
@app.route('/suggest_pin_boards', methods=['GET', 'POST'])
def suggest_pin_boards():
    if 'user_id' in session:
        user_id = session['user_id']
        current_user = User.query.get(user_id)
        suggested_pin_boards = current_user.suggested_pin_boards()

        return render_template('suggest_pin_boards.html', suggested_pin_boards=suggested_pin_boards)
###ADMIN##

@app.route('/admin/')
def index_admin():

    if 'user_id' in session:
        # User is logged in
        staff_count = User.query.filter_by(type=1).count()

        # Count the number of regular users
        regular_user_count = User.query.filter_by(type=2).count()

        # Count the total number of images
        total_images = Image.query.count()
        image_data = Image.query.with_entities(Image.timestamp).all()

        # Extract timestamps and count images for each date
        date_counts = {}
        for timestamp in image_data:
            date_str = timestamp[0].strftime('%Y-%m-%d')
            if date_str in date_counts:
                date_counts[date_str] += 1
            else:
                date_counts[date_str] = 1

        # Prepare data for the chart
        labels = list(date_counts.keys())
        data = list(date_counts.values())
        return render_template("admin/index.html", staff_count=staff_count, regular_user_count=regular_user_count,
                               total_images=total_images,labels=labels,data=data)
    else:
        # User is not logged in, redirect to the login page
        return redirect(url_for('login_admin'))
@app.route('/admin/login/', methods=['GET', 'POST'])
def login_admin():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = User.query.filter_by(username=username).first()

        if user:
            if bcrypt.check_password_hash(user.password, password):
                # Login successful
                session['user_id'] = user.id
                session['username'] = user.username
                session['type'] = user.type
                session['avatar'] = user.avatar
                if user.type ==0 or user.type==1:
                    return redirect(url_for('index_admin'))

            else:
                # Login failed
                return "Login failed. Passwords do not match."

            # Handle the case where the username does not exist
        return "Login failed. User not found."

    return render_template('admin/login.html')
@app.route('/admin/logout')
def logout_admin():
    # Clear the user session
    session.pop('user_id', None)
    session.pop('username', None)
    session.pop('type', None)

    # Redirect to the login page
    return redirect(url_for('login_admin'))
@app.route('/admin/staffadd/', methods=['GET','POST'])
def staffadd():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
    # Check if the post request has the file part
        if 'avatar' not in request.files:
            return 'No file part'

        avatar = request.files['avatar']
        type=1

    # If the user does not select a file, the browser submits an empty part without filename
        if avatar.filename == '':
            return 'No selected file'

        if avatar and allowed_file(avatar.filename):
        # Save the avatar file to the upload folder
            filename = secure_filename(avatar.filename)
            avatar_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            avatar.save(avatar_path)

        # Create a new user with the avatar path
            new_user = User(username=username, password=hashed_password, email=email, avatar=filename,type=type)
            db.session.add(new_user)
            db.session.commit()
            flash('User created successfully', 'success')
            return redirect(url_for('staffadd'))


    return render_template("admin/staffadd.html")

@app.route('/admin/display_staff', methods=['GET'])
def display_staff():
    # Assuming you have a 'User' model with a 'type' column
    users = User.query.filter_by(type=1).all()

    return render_template("admin/display_staff.html", users=users)
@app.route('/admin/change_password/<int:user_id>', methods=['GET', 'POST'])
def change_password_admin(user_id):
    user = User.query.get(user_id)

    if not user:
        flash('User not found', 'error')
        return redirect(url_for('display_staff'))

    if request.method == 'POST':
        new_password = request.form.get('new_password')
        hashed_password = bcrypt.generate_password_hash(new_password).decode('utf-8')

        # Update the user's hashed password
        user.password = hashed_password
        db.session.commit()

        flash('Password changed successfully', 'success')
        return redirect(url_for('display_staff'))

    return render_template('admin/change_password.html', user=user)
@app.route('/admin/delete_staff/<int:user_id>', methods=['GET'])
def delete_staff(user_id):
    user = User.query.get(user_id)

    if not user:
        flash('User not found', 'error')
        return redirect(url_for('display_staff'))

    # Delete the user from the database
    db.session.delete(user)
    db.session.commit()

    flash('Staff member deleted successfully', 'success')
    return redirect(url_for('display_staff'))
@app.route('/admin/display_images')
def display_images_admin():
    # Assuming you have a 'User' model with a 'type' column
    # Assuming you have a 'Image' model with a 'user_id' column
    # Assuming 'type=2' is for regular users
    users = User.query.filter_by(type=2).all()

    # Collect images for all regular users
    images = []
    for user in users:
        user_images = Image.query.filter_by(user_id=user.id).all()
        images.extend(user_images)

    return render_template("admin/display_images.html", images=images)


@app.route('/admin/images/delete/<int:image_id>', methods=['POST'])
def admin_delete_image(image_id):
    image = Image.query.get_or_404(image_id)

    try:
        # Xóa ảnh từ cơ sở dữ liệu
        db.session.delete(image)
        db.session.commit()

        # Thông báo xóa thành công
        flash('Image deleted successfully', 'success')
    except:
        # Nếu có lỗi, hủy bỏ các thay đổi và hiển thị thông báo lỗi
        db.session.rollback()
        flash('Error deleting image', 'error')

    return redirect(url_for('display_images_admin'))
# Add a new route to display comments for a specific image
@app.route('/admin/image_comments/<int:image_id>')
def image_comments_admin(image_id):
    # Retrieve the image by its ID
    image = Image.query.get(image_id)

    if not image:
        flash('Image not found', 'error')
        return redirect(url_for('index'))

    # Retrieve comments for the image
    comments = Comment.query.filter_by(image_id=image.id).all()

    return render_template("admin/image_comments.html", image=image, comments=comments)
# Add a new route to display users with type=2
@app.route('/admin/display_regular_users')
def display_regular_users_admin():
    # Assuming you have a 'User' model with a 'type' column
    users = User.query.filter(User.type.notin_([0, 1])).order_by(User.id.desc()).all()

    return render_template("admin/display_regular_users.html", users=users)
# Add a new route to display images for a specific user
@app.route('/admin/user_images/<int:user_id>')
def user_images_admin(user_id):
    # Retrieve the user by their ID
    user = User.query.get(user_id)

    if not user:
        flash('User not found', 'error')
        return redirect(url_for('index'))

    # Retrieve images for the user
    images = Image.query.filter_by(user_id=user.id).order_by(Image.id.desc()).all()

    return render_template("admin/user_images.html", user=user, images=images)
# Add a new route to display image details
@app.route('/admin/image_details/<int:image_id>')
def image_details_admin(image_id):
    # Retrieve the image by its ID
    image = Image.query.get(image_id)

    if not image:
        flash('Image not found', 'error')
        return redirect(url_for('index'))

    return render_template("admin/image_details.html", image=image)
@app.route('/admin/reported_images', methods=['GET'])
def reported_images_admin():
    # Query reported images from the database
    reported_images = ReportImage.query.all()

    # Render the template with reported images
    return render_template('admin/reported_images.html', reported_images=reported_images)
@app.route('/admin/block_user/<int:user_id>', methods=['GET'])
def block_user_admin(user_id):
    # Query reported images from the database
    user = User.query.get(user_id)
    user.type = 3 if user.type == 2 else 2
    db.session.commit()
    # Render the template with reported images
    return redirect(url_for('display_regular_users_admin'))
@app.route('/admin/block_user_mail/<int:user_id>', methods=['GET'])
def block_user_admin_mail(user_id):
    # Query user from the database
    user = User.query.get(user_id)

    # Update user type
    user.type = 3 if user.type == 2 else 2
    db.session.commit()

    # Send email based on user type
    if user.type == 3:
        send_blocked_email(user.email)
    elif user.type == 2:
        send_unblocked_email(user.email)

    # Redirect to the page displaying regular users
    return redirect(url_for('display_regular_users_admin'))
@app.route('/admin/hashtag', methods=['GET'])
def admin_hashtag():
    # Query reported images from the database
    unique_hashtags = db.session.query(distinct(Hashtag.name)).all()

    # Chuyển đổi kết quả từ list các tuple sang list các string
    unique_hashtags = [tag[0] for tag in unique_hashtags]

    # Render the template with reported images
    return render_template('admin/hashtag.html', hashtag=unique_hashtags)
@app.route('/admin/hashtag/<string:hashtag_name>', methods=['GET'])
def view_hashtag(hashtag_name):
    # Tìm kiếm theo hashtag
    images_with_hashtag = db.session.query(Image). \
        join(ImageHashtags, Image.id == ImageHashtags.image_id). \
        join(Hashtag, Hashtag.id == ImageHashtags.hashtag_id). \
        filter(Hashtag.name == hashtag_name).all()

    return render_template('admin/view_hashtag.html', hashtag=hashtag_name, images_with_hashtag=images_with_hashtag)
if __name__ == '__main__':
    app.run(debug=True)
