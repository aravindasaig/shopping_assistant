from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, joinedload
from schema import *

engine = create_engine("sqlite:///retail.db")
Session = sessionmaker(bind=engine)
session = Session()

def print_tshirts():
    print("\n🧵 T-SHIRTS")
    results = (
        session.query(Product)
        .join(ProductType).join(Subcategory).join(Category)
        .join(TShirtAttributes)
        .options(joinedload(Product.product_type).joinedload(ProductType.subcategory).joinedload(Subcategory.category))
        .all()
    )

    for p in results:
        attrs = session.query(TShirtAttributes).filter_by(product_id=p.product_id).first()
        print(f"📦 {p.product_name} | {p.brand} | ₹{p.price_inr}")
        print(f"  - Category: {p.product_type.subcategory.category.category_name}")
        print(f"  - Subcategory: {p.product_type.subcategory.subcategory_name}")
        print(f"  - Type: {p.product_type.product_type_name}")
        print(f"  - Color: {attrs.color}, Pattern: {attrs.pattern}, Neck: {attrs.neck_type}, Tags: {attrs.visual_tags}")
        print()

def print_tvs():
    print("\n📺 TELEVISIONS")
    results = (
        session.query(Product)
        .join(ProductType).join(Subcategory).join(Category)
        .join(TVAttributes)
        .options(joinedload(Product.product_type).joinedload(ProductType.subcategory).joinedload(Subcategory.category))
        .all()
    )

    for p in results:
        attrs = session.query(TVAttributes).filter_by(product_id=p.product_id).first()
        print(f"📦 {p.product_name} | {p.brand} | ₹{p.price_inr}")
        print(f"  - Category: {p.product_type.subcategory.category.category_name}")
        print(f"  - Subcategory: {p.product_type.subcategory.subcategory_name}")
        print(f"  - Type: {p.product_type.product_type_name}")
        print(f"  - Screen: {attrs.screen_size}, Resolution: {attrs.resolution}, Smart: {attrs.smart_tv}, Ports: {attrs.ports}")
        print()

if __name__ == "__main__":
    print_tshirts()
    print_tvs()
