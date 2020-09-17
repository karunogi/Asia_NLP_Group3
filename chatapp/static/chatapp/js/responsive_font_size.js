(function(){
	$(document).ready(function(){
			$('.responsive-font-size').each(function(){
				$(this).css({'font-size':($(window).width())*($(this).data('font-ratio'))*0.006+'em'});
			});
	});
})();

$(document).ready(function(){
		$(window).resize(function(){
				$('.responsive-font-size').each(function(){
						$(this).css({'font-size':($(window).width())*($(this).data('font-ratio'))*0.006+'em'});
				});
		});
});
