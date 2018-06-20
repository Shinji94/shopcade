db.getCollection('trkelle_view').aggregate([
{$match:{
   $and: [  
    { "tag": {'$in':['dynamic-product-view','list-product-view','product-product-view','button-product-view']}},
    {'data':{'$in':['5a9198c228c13329f58d372c']}}
]}},
{     $group: {
      _id: '$scid', 
      count: {
        $sum: 1
      }
    }
  }
])